from pathlib import Path
from tempfile import TemporaryDirectory

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from configuration import chevauchementChunk
from configuration import extensionsAcceptees
from configuration import tailleChunk
from ressources import obtenirBaseVectorielle


def chargerDocument(cheminFichier: Path, nomSource: str) -> list[Document]:
    extension = cheminFichier.suffix.lower()

    if extension == ".pdf":
        chargeur = PyMuPDFLoader(str(cheminFichier))
        documents = chargeur.load()
    elif extension == ".txt":
        chargeur = TextLoader(
            str(cheminFichier),
            encoding="utf-8",
            autodetect_encoding=True,
        )
        documents = chargeur.load()
    elif extension == ".md":
        contenu = cheminFichier.read_text(encoding="utf-8")
        document = Document(page_content=contenu)
        documents = [document]
    else:
        raise ValueError("Format non pris en charge : " + extension)

    documentsValides = []

    for document in documents:
        contenu = document.page_content.strip()

        if contenu:
            document.metadata["source"] = nomSource
            document.metadata["typeFichier"] = extension.lstrip(".")
            documentsValides.append(document)

    if not documentsValides:
        raise ValueError("Le fichier ne contient aucun texte exploitable.")

    return documentsValides


def chargerFichiersTeleverses(fichiersTeleverses):
    documents = []
    erreurs = []
    sourcesChargees = []

    with TemporaryDirectory() as repertoireTemporaire:
        cheminTemporaire = Path(repertoireTemporaire)

        for position, fichierTeleverse in enumerate(fichiersTeleverses):
            nomSource = Path(fichierTeleverse.name).name
            extension = Path(nomSource).suffix.lower()

            if extension not in extensionsAcceptees:
                message = nomSource + " : format non pris en charge."
                erreurs.append(message)
                continue

            try:
                nomTemporaire = str(position) + "_" + nomSource
                cheminFichier = cheminTemporaire / nomTemporaire
                cheminFichier.write_bytes(fichierTeleverse.getvalue())

                documentsFichier = chargerDocument(cheminFichier, nomSource)
                documents.extend(documentsFichier)
                sourcesChargees.append(nomSource)
            except Exception as erreur:
                message = nomSource + " : " + str(erreur)
                erreurs.append(message)

    return documents, sourcesChargees, erreurs


def decouperDocuments(documents: list[Document]) -> list[Document]:
    separations = ["\n\n", "\n", ". ", " ", ""]

    decoupeur = RecursiveCharacterTextSplitter(
        chunk_size=tailleChunk,
        chunk_overlap=chevauchementChunk,
        separators=separations,
        length_function=len,
        add_start_index=True,
    )

    chunks = decoupeur.split_documents(documents)

    for position, chunk in enumerate(chunks):
        chunk.metadata["numeroChunk"] = position

    return chunks


def indexerFichiers(fichiersTeleverses):
    if not fichiersTeleverses:
        raise ValueError("Sélectionnez au moins un fichier avant l'indexation.")

    resultatChargement = chargerFichiersTeleverses(fichiersTeleverses)
    documents = resultatChargement[0]
    sourcesChargees = resultatChargement[1]
    erreurs = resultatChargement[2]

    if not documents:
        details = " ".join(erreurs)

        if not details:
            details = "Aucun contenu n'a été extrait."

        raise ValueError(details)

    chunks = decouperDocuments(documents)

    if not chunks:
        raise ValueError("Aucun chunk n'a pu être créé.")

    baseVectorielle = obtenirBaseVectorielle()
    baseVectorielle.reset_collection()
    baseVectorielle.add_documents(chunks)

    nombreFichiers = len(sourcesChargees)
    nombreChunks = len(chunks)

    return nombreFichiers, nombreChunks, sourcesChargees, erreurs

