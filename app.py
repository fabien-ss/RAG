import streamlit as st

from configuration import nomModeleOllama
from ingestion import indexerFichiers
from interface import afficherHistorique
from interface import afficherResultatsSemantiques
from interface import afficherSourcesRag
from interface import initialiserEtatSession
from recherche import effectuerRechercheSemantique
from recherche import genererReponseRag


def afficherBarreLaterale():
    with st.sidebar:
        st.title("📚 Assistant RAG local")
        st.caption("Interrogez vos documents sans utiliser d'API externe.")

        st.subheader("Documents")
        fichiersTeleverses = st.file_uploader(
            "Charger des fichiers",
            type=["pdf", "md", "txt"],
            accept_multiple_files=True,
            help="Formats acceptés : PDF, Markdown et TXT.",
        )

        boutonIndexation = st.button(
            "Indexer",
            type="primary",
            use_container_width=True,
        )

        if boutonIndexation:
            executerIndexation(fichiersTeleverses)

        if st.session_state.indexPret:
            nombreFichiers = len(st.session_state.fichiersIndexes)
            nombreChunks = st.session_state.nombreChunks

            messageIndex = (
                "Index actif : "
                + str(nombreFichiers)
                + " fichier(s), "
                + str(nombreChunks)
                + " chunk(s)."
            )
            st.caption(messageIndex)

        st.divider()
        llmActif = st.toggle(
            "Activer le LLM",
            key="llmActif",
        )

        if llmActif:
            st.success("Mode actif : Assistant RAG")
        else:
            st.info("Mode actif : Recherche sémantique")

    return llmActif


def executerIndexation(fichiersTeleverses) -> None:
    try:
        with st.spinner("Extraction, découpage et vectorisation..."):
            resultat = indexerFichiers(fichiersTeleverses)

        nombreFichiers = resultat[0]
        nombreChunks = resultat[1]
        sourcesChargees = resultat[2]
        erreurs = resultat[3]

        st.session_state.indexPret = True
        st.session_state.fichiersIndexes = sourcesChargees
        st.session_state.nombreChunks = nombreChunks

        messageSucces = (
            "Indexation terminée : "
            + str(nombreFichiers)
            + " fichier(s), "
            + str(nombreChunks)
            + " chunk(s)."
        )
        st.success(messageSucces)

        for erreur in erreurs:
            st.warning(erreur)
    except ValueError as erreur:
        st.warning(str(erreur))
    except Exception as erreur:
        message = "L'indexation a échoué. Détail : " + str(erreur)
        st.error(message)


def enregistrerQuestion(question: str) -> None:
    message = {
        "role": "user",
        "contenu": question,
    }
    st.session_state.messages.append(message)

    with st.chat_message("user"):
        st.markdown(question)


def traiterRechercheSemantique(question: str) -> None:
    try:
        with st.spinner("Recherche des passages similaires..."):
            resultats = effectuerRechercheSemantique(question)

        message = {
            "role": "assistant",
            "mode": "semantique",
            "resultats": resultats,
        }
        st.session_state.messages.append(message)

        with st.chat_message("assistant"):
            afficherResultatsSemantiques(resultats)
    except Exception as erreur:
        message = "La recherche sémantique a échoué : " + str(erreur)
        st.error(message)


def traiterQuestionRag(question: str) -> None:
    try:
        with st.spinner("Recherche et génération de la réponse..."):
            resultat = genererReponseRag(question)

        reponse = resultat[0]
        sources = resultat[1]

        message = {
            "role": "assistant",
            "mode": "rag",
            "contenu": reponse,
            "sources": sources,
        }
        st.session_state.messages.append(message)

        with st.chat_message("assistant"):
            st.markdown(reponse)
            afficherSourcesRag(sources)
    except Exception as erreur:
        message = (
            "Le mode RAG a échoué. Vérifiez qu'Ollama est démarré et que le modèle '"
            + nomModeleOllama
            + "' est installé. Détail : "
            + str(erreur)
        )
        st.error(message)


def traiterQuestion(question: str, llmActif: bool) -> None:
    enregistrerQuestion(question)

    if llmActif:
        traiterQuestionRag(question)
    else:
        traiterRechercheSemantique(question)


def lancerApplication() -> None:
    st.set_page_config(
        page_title="Assistant RAG local",
        page_icon="📚",
        layout="centered",
    )

    initialiserEtatSession()
    llmActif = afficherBarreLaterale()

    st.title("Assistant documentaire local")
    st.caption("Posez une question sur les documents indexés.")

    afficherHistorique()

    question = st.chat_input("Posez une question sur vos documents")

    if question:
        traiterQuestion(question, llmActif)


if __name__ == "__main__":
    lancerApplication()
