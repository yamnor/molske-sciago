import threading

#import os
#os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

import av
import cv2
from PIL import Image
from io import BytesIO

import torch
import numpy as np
import pandas as pd
import math
import re

from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors, rdMolDescriptors
from rdkit.Chem.Draw import SimilarityMaps, rdMolDraw2D

import py3Dmol
from stmol import showmol

atomcolor = { # HSV
  'C'  : {'min' : (  0,   0,   0), 'max' : (180, 255,  40)},
  'N'  : {'min' : ( 90,  64,   0), 'max' : (150, 255, 255)},
  'O1' : {'min' : (150,  64,   0), 'max' : (180, 255, 255)},
  'O2' : {'min' : (  0,  64,   0), 'max' : ( 30, 255, 255)}}

bondtype = {
  'none'   : 0,
  'single' : 1,
  'double' : 2,
  'triple' : 3}

def draw_text(img, txt, xy):
  face = cv2.FONT_HERSHEY_PLAIN
  size = 2
  thickness = 3
  color = (255, 255, 255)
  (w, h), _ = cv2.getTextSize(txt, face, size, thickness)
  org = [int(xy[0] - w / 2), int(xy[1] + h / 2)]
  cv2.putText(img, txt, org, face, size, color, thickness, cv2.LINE_AA)

def draw_bond(img, atom, adjmat):
  natoms = len(atom)
  colors = [
    (255, 255, 255),  # none
    (  0, 255,   0),  # single
    (  0,   0, 255),  # double
    (  0, 255, 255)]  # triple
  for i in range(natoms):
    for j in range(i + 1, natoms):
      xi, yi = atom[i]['geom']
      xj, yj = atom[j]['geom']
      dx = - (yj - yi) / math.sqrt(((yj - yi))**2 + ((xj - xi))**2)
      dy =   (xj - xi) / math.sqrt(((yj - yi))**2 + ((xj - xi))**2)
      aij = int(adjmat[i, j])
      if aij == 1:
        cv2.line(img, (xi, yi), (xj, yj), colors[aij], thickness=2, lineType=cv2.LINE_AA)
      elif aij == 2:
        scale = 5
        dx, dy = int(dx * scale), int(dy * scale)
        xi_u, yi_u, xi_d, yi_d = dx + xi, dy + yi, - dx + xi, - dy + yi
        xj_u, yj_u, xj_d, yj_d = dx + xj, dy + yj, - dx + xj, - dy + yj
        cv2.line(img, (xi_u, yi_u), (xj_u, yj_u), colors[aij], thickness=2, lineType=cv2.LINE_AA)
        cv2.line(img, (xi_d, yi_d), (xj_d, yj_d), colors[aij], thickness=2, lineType=cv2.LINE_AA)
      elif aij == 3:
        scale = 10
        dx, dy = int(dx * scale), int(dy * scale)
        xi_u, yi_u, xi_d, yi_d = dx + xi, dy + yi, - dx + xi, - dy + yi
        xj_u, yj_u, xj_d, yj_d = dx + xj, dy + yj, - dx + xj, - dy + yj
        cv2.line(img, (xi_u, yi_u), (xj_u, yj_u), colors[aij], thickness=2, lineType=cv2.LINE_AA)
        cv2.line(img, (xi, yi), (xj, yj), colors[aij], thickness=2, lineType=cv2.LINE_AA)
        cv2.line(img, (xi_d, yi_d), (xj_d, yj_d), colors[aij], thickness=2, lineType=cv2.LINE_AA)

def draw_label(img, item):
  nitems = len(item)
  for i in range(nitems):
    draw_text(img, item[i]['name'], item[i]['geom'])

def adj2mol(atom, adjmat):
  natoms = len(atom)
  mol = Chem.RWMol()
  idx = {}
  for i in range(natoms):
    idx[i] = mol.AddAtom(Chem.Atom(atom[i]))
  for i in range(natoms):
    for j in range(i + 1, natoms):
      aij = adjmat[i, j]
      if aij == 0:
        continue
      elif aij == 1:
        mol.AddBond(idx[i], idx[j], Chem.rdchem.BondType.SINGLE)
      elif aij == 2:
        mol.AddBond(idx[i], idx[j], Chem.rdchem.BondType.DOUBLE)
      elif aij == 3:
        mol.AddBond(idx[i], idx[j], Chem.rdchem.BondType.TRIPLE)
  return mol.GetMol()

def smi2mol(smi):
  mol = Chem.MolFromSmiles(smi)
  if mol is not None:
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol, maxIters = 200)
    return mol
  else:
    return None

def img2smi(img, model):

  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  obj = model(gray).pandas().xyxy[0]

  atom = []
  for i in range(len(obj.name)):
    if obj.name[i] == 'atom':
      xmin = int(obj.xmin[i])
      ymin = int(obj.ymin[i])
      xmax = int(obj.xmax[i])
      ymax = int(obj.ymax[i])
      w = xmax - xmin
      h = ymax - ymin
      cx = int(xmin + w / 2)
      cy = int(ymin + h / 2)
      crop_xmin = int(xmin + w * 0.2)
      crop_ymin = int(ymin + h * 0.2)
      crop_xmax = int(xmax - w * 0.2)
      crop_ymax = int(ymax - h * 0.2)
      crop_w = crop_xmax - crop_xmin
      crop_h = crop_ymax - crop_ymin
      cropped = hsv[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
      likelihood = {'C' : 0.0, 'N': 0.0, 'O' : 0.0}
      for a in ['C', 'N', 'O1', 'O2']:
        mask = cv2.inRange(cropped, atomcolor[a]['min'], atomcolor[a]['max'])
        likelihood[a] = sum(mask.flatten()) / 255 / (crop_w * crop_h)
      if max(likelihood.values()) > 0.5:
        name = max(likelihood, key = likelihood.get)
        name = re.sub(r'[0-9]+', '', name)
        geom = np.array([cx, cy])
        atom.append({'name' : name, 'geom' : geom})

  natoms = len(atom)

  bond = []
  for i in range(len(obj.name)):
    if obj.name[i] != 'atom':
      xmin = int(obj.xmin[i])
      ymin = int(obj.ymin[i])
      xmax = int(obj.xmax[i])
      ymax = int(obj.ymax[i])
      cx = int((xmin + xmax) / 2)
      cy = int((ymin + ymax) / 2)
      name = obj.name[i]
      geom = np.array([cx, cy])
      bond.append({'name' : name, 'geom' : geom})

  nbonds = len(bond)

  dismat = np.zeros((natoms, natoms))
  mindis = 1000.0
  for i in range(natoms):
    for j in range(i + 1, natoms):
      xi, yi = atom[i]['geom']
      xj, yj = atom[j]['geom']
      dismat[i, j] = math.sqrt((xi - xj)**2 + (yi - yj)**2)
      dismat[j, i] = dismat[i, j]
      if dismat[i, j] < mindis:
        mindis = dismat[i, j]

  adjmat = np.zeros((natoms, natoms))
  atomlist = []
  for i in range(natoms):
    atomlist.append(atom[i]['name'])
    for j in range(i + 1, natoms):
      xi, yi = atom[i]['geom']
      xj, yj = atom[j]['geom']
      cx = int((xi + xj) / 2)
      cy = int((yi + yj) / 2)
      if dismat[i, j] < mindis * 1.5:
        aij = 0
        for k in range(nbonds):
          xk, yk = bond[k]['geom']
          if (cx - mindis / 4) < xk < (cx + mindis / 4) and (cy - mindis / 4) < yk < (cy + mindis / 4):
            aij = bondtype[bond[k]['name']]
        adjmat[i, j] = aij
        adjmat[j, i] = aij

  draw_bond(img, atom, adjmat)
  draw_label(img, atom)

  return Chem.MolToSmiles(adj2mol(atomlist, adjmat))

def show_2dview(smi):
  mol = Chem.MolFromSmiles(smi)
  if mol is not None:
    st.image(Draw.MolToImage(mol))
  else:
    st.error('Try again.')

def show_3dview(smi):
  viewsize = (300, 300)
  mol = smi2mol(smi)
  if mol is not None:
    viewer = py3Dmol.view(height = viewsize[0], width = viewsize[1])
    molblock = Chem.MolToMolBlock(mol)
    viewer.addModel(molblock, 'mol')
    viewer.setStyle({'stick':{}})
    viewer.zoomTo()
    viewer.spin('y', 1)
    showmol(viewer, height = viewsize[0], width = viewsize[1])
    st.balloons()
  else:
    st.error('Try again.')

def show_properties(smi):
  mol = Chem.MolFromSmiles(smi)
  if mol is not None:
    col = st.columns(3)
    col[0].metric(label = "Mol. Weight",    value = '{:.1f}'.format(Descriptors.MolWt(mol)))
    #col[1].metric(label = "Hetero Atoms",        value = Descriptors.NumHeteroatoms(mol))
    col[1].metric(label = "sp3 C Frac.", value = '{:.2f}'.format(Descriptors.FractionCSP3(mol)))
    col[2].metric(label = "Rot. Bonds",     value = Descriptors.NumRotatableBonds(mol))
    col = st.columns(3)
    col[0].metric(label = "LogP",                value = '{:.2f}'.format(Descriptors.MolLogP(mol)))
    #col[1].metric(label = "Polar Surface Area",  value = '{:.1f}'.format(Descriptors.TPSA(mol)))
    col[1].metric(label = "H-bond Accep",    value = Descriptors.NumHAcceptors(mol))
    col[2].metric(label = "H-bond Donor",       value = Descriptors.NumHDonors(mol))

    if mol.GetNumHeavyAtoms() > 1:

      st.markdown('---')

      st.subheader('Charge Map')
      AllChem.ComputeGasteigerCharges(mol)
      contribs = [mol.GetAtomWithIdx(i).GetDoubleProp('_GasteigerCharge') for i in range(mol.GetNumAtoms())]    
      view = rdMolDraw2D.MolDraw2DSVG(300, 300)
      SimilarityMaps.GetSimilarityMapFromWeights(mol, contribs, colorMap='bwr', draw2d = view)
      view.FinishDrawing()
      st.image(view.GetDrawingText())

      st.markdown('---')

      st.subheader('LogP Map')
      contribs = [x for x, y in rdMolDescriptors._CalcCrippenContribs(mol)]
      view = rdMolDraw2D.MolDraw2DSVG(300, 300)
      SimilarityMaps.GetSimilarityMapFromWeights(mol, contribs, colorMap='PRGn', draw2d = view)
      view.FinishDrawing()
      st.image(view.GetDrawingText())

      st.markdown('---')

      st.subheader('Polar Surface Area Map')
      contribs = rdMolDescriptors._CalcTPSAContribs(mol)
      view = rdMolDraw2D.MolDraw2DSVG(300, 300)
      SimilarityMaps.GetSimilarityMapFromWeights(mol, contribs, colorMap='RdBu', draw2d = view)
      view.FinishDrawing()
      st.image(view.GetDrawingText())

      st.markdown('---')

      st.subheader('Accessible Surface Area Map')
      contribs = rdMolDescriptors._CalcLabuteASAContribs(mol)
      contribs = [x for x in contribs[0]]
      view = rdMolDraw2D.MolDraw2DSVG(300, 300)
      SimilarityMaps.GetSimilarityMapFromWeights(mol, contribs, colorMap='RdBu', draw2d = view)
      view.FinishDrawing()
      st.image(view.GetDrawingText())

  else:
    st.error('Try again.')

def main():

  st.set_page_config(
    page_title = 'molske',
    #page_icon = 'logo2.png',
    initial_sidebar_state = 'auto')

  class VideoProcessor:
    frame_lock: threading.Lock
    smi: None

    def __init__(self) -> None:
      self.frame_lock = threading.Lock()
      self.smi = None
      self.model = torch.hub.load('ultralytics/yolov5', 'custom', path = 'model/molske.pt')

    def recv(self, frame):
      img = frame.to_ndarray(format="bgr24")
      smi = img2smi(img, self.model)
      with self.frame_lock:
        self.smi = smi
      return av.VideoFrame.from_ndarray(img, format="bgr24")

  # st.title("molske ✍️")

  ctx = webrtc_streamer(
    key = "webrtc",
    media_stream_constraints = {"video": True, "audio": False},
    video_processor_factory = VideoProcessor,
    rtc_configuration = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

  with st.sidebar:

    st.title("molske ✍️")

    if ctx.video_processor:

      st.markdown("Click 😊, if you are satisfied with the detected chemical structure of your molecule.")

      if st.button("😊"):

        with ctx.video_processor.frame_lock:
          smi = ctx.video_processor.smi

          if smi is not None:
            st.markdown("---")

            st.subheader('SMILES')
            st.code(smi)
            st.markdown(
              """
              [SMILES](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system)
              is a specification for describing the chemical structure of molecules using short strings.
              """)

            st.markdown("---")

            st.subheader('2D Structure')
            show_2dview(smi)

            st.markdown("---")

            st.subheader('3D Structure')
            show_3dview(smi)

            st.markdown("---")

            st.subheader('Properties')
            show_properties(smi)

          else:
            st.warning("No frames available yet.")

    st.markdown('---')

    st.subheader('How to play')

    # st.video('https://user-images.githubusercontent.com/134783/185785833-f35cfd95-e499-4cc5-bf1f-54623453075d.mp4', format="video/mp4")

    st.markdown(
      """
      A molecule sketched with _hexagon-shaped_ **atoms** and _hand-drawn_ **bonds** can be detected
      with a web camera to convert it into 2D & 3D structures, and predict its chemical properties.

      * **Black**, **blue**, and **red** hexagon-shaped parts are recognized as
        carbon (**C**), nitrogen (**N**), and oxygen (**O**) atoms.
      * **Green**, **red**, and **yellow** lines shown on the video screen represent
        **single**, **double**, and **triple** bonds.
      * Chemical structures can be drawn in [skeletal formula](https://en.wikipedia.org/wiki/Skeletal_formula).
        Hydrogen (**H**) atoms are automatically added according to the detected chemical structure.
      """)

    st.subheader("Notice")

    st.markdown(
      """
      * This web app is hosted on a cloud server ([Streamlit Cloud](https://streamlit.io/))
         and videos are sent to the server for processing.
      * No data is stored, everything is processed in memory and discarded,
        but if this is a concern for you, please refrain from using this app.
      """)

    st.subheader("Author")

    st.markdown(
      """
      * This web app is developed by Yamamoto, Norifumi. [![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40yamnor)](https://twitter.com/yamnor)
      * You can follow me on social media:
        [GitHub](https://github.com/yamnor) | 
        [LinkedIn](https://www.linkedin.com/in/yamnor) | 
        [WebSite](https://yamlab.net).
      """)
    
    st.caption("[molske science agora ver.](https://github.com/yamnor/molske-sciago)")

if __name__ == "__main__":
    main()
