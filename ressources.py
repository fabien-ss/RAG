import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

from configuration import nomCollection
from configuration import nomModeleEmbeddings
from configuration import nomModeleOllama
from configuration import repertoireChroma


@st.cache_resource
def obtenirModeleEmbeddings() -> HuggingFaceEmbeddings:
    modele = HuggingFaceEmbeddings(
        model_name=nomModeleEmbeddings,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return modele


@st.cache_resource
def obtenirBaseVectorielle() -> Chroma:
    baseVectorielle = Chroma(
        collection_name=nomCollection,
        embedding_function=obtenirModeleEmbeddings(),
        persist_directory=repertoireChroma,
    )
    return baseVectorielle


@st.cache_resource
def obtenirModeleOllama() -> ChatOllama:
    modele = ChatOllama(
        model=nomModeleOllama,
        temperature=0,
    )
    return modele

