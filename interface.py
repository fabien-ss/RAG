import streamlit as st

from ressources import obtenirBaseVectorielle


def verifierPresenceIndex() -> bool:
    try:
        baseVectorielle = obtenirBaseVectorielle()
        donnees = baseVectorielle.get(limit=1)
        identifiants = donnees.get("ids", [])

        if identifiants:
            return True

        return False
    except Exception:
        return False


def initialiserEtatSession() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "baseVectorielle" not in st.session_state:
        st.session_state.baseVectorielle = obtenirBaseVectorielle()

    if "indexPret" not in st.session_state:
        st.session_state.indexPret = verifierPresenceIndex()

    if "fichiersIndexes" not in st.session_state:
        st.session_state.fichiersIndexes = []

    if "nombreChunks" not in st.session_state:
        st.session_state.nombreChunks = 0


def afficherResultatsSemantiques(resultats: list[dict]) -> None:
    if not resultats:
        st.warning("Aucun résultat. Indexez d'abord un document.")
        return

    for position, resultat in enumerate(resultats, start=1):
        with st.container(border=True):
            source = resultat["source"]
            contenu = resultat["contenu"]
            st.caption("Résultat " + str(position) + " — Source : " + source)
            st.text(contenu)


def afficherSourcesRag(sources: list[dict]) -> None:
    with st.expander("Afficher les extraits utilisés comme contexte"):
        if not sources:
            st.info("Aucun extrait n'a été trouvé.")
            return

        for position, sourceDocument in enumerate(sources, start=1):
            nomSource = sourceDocument["source"]
            contenu = sourceDocument["contenu"]

            titre = "**Extrait " + str(position) + " — Source : " + nomSource + "**"
            st.markdown(titre)
            st.text(contenu)

            if position < len(sources):
                st.divider()


def afficherHistorique() -> None:
    for message in st.session_state.messages:
        role = message["role"]

        with st.chat_message(role):
            if role == "user":
                st.markdown(message["contenu"])
                continue

            mode = message["mode"]

            if mode == "rag":
                st.markdown(message["contenu"])
                afficherSourcesRag(message["sources"])
            else:
                afficherResultatsSemantiques(message["resultats"])

