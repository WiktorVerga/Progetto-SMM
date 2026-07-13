import os
import re
from pathlib import Path

import pandas as pd

try:
    import orjson as json
except ImportError:
    import json


# Cartella di input: contiene una sottocartella per subreddit, ognuna con i
# dump .jsonl dei post e dei commenti di quel subreddit.
RAW_DATA_FOLDER = r"D:\UNI\Corsi\Social Media Mining\Progetto SMM\data\rawdata"

# Keywords da cercare (case-insensitive, parola intera o sottostringa)
KEYWORDS = [
    "AI",
    "artificial intelligence",
    "machine learning",
    "ChatGPT",
    "Copilot",
    "Midjourney",
    "Stable Diffusion",
    "automation",
    "job loss",
    "replace",
    "workflow",
    "generative AI",
    "LLM",
    "prompt",
]

# Autori "bot" da escludere (post/commenti automatici, non opinioni reali)
BOT_AUTHORS = {"AutoModerator"}

# Valori che indicano contenuto non più disponibile
DELETED_MARKERS = {"[deleted]", "[removed]"}

#  CAMPI CSV

POST_FIELDS = [
    "id",  # ID univoco del post (es. "abc123")
    "author",  # Username autore
    "subreddit",  # Nome subreddit (senza r/)
    "title",  # Titolo del post
    "selftext",  # Testo del corpo del post
    "matched_keywords",  # Keywords trovate nel testo (debug/analisi)
]

COMMENT_FIELDS = [
    "id",  # ID univoco del commento
    "link_id",  # ID del post padre (formato "t3_xxxxx")
    "subreddit",  # Nome subreddit da cui è stato recuperato il commento
    "author",  # Username autore
    "body",  # Testo del commento
]

# Pattern regex compilato una volta sola per performance
_keyword_pattern = re.compile(
    r'(?<!\w)(' + '|'.join(re.escape(kw) for kw in KEYWORDS) + r')(?!\w)',
    re.IGNORECASE
)


#  FUNZIONI DI SUPPORTO

def _find_keywords(text: str) -> str:
    """Restituisce le keyword trovate nel testo, separate da '|'."""
    if not text:
        return ""
    found = set(m.group(0).lower() for m in _keyword_pattern.finditer(text))
    return "|".join(sorted(found))


def _text_matches(text: str) -> bool:
    """True se almeno una keyword è presente nel testo."""
    return bool(text and _keyword_pattern.search(text))


def _read_jsonl(path: Path):
    """Legge un file .jsonl riga per riga e restituisce i dict corrispondenti."""
    with open(path, "rb") as f:
        for line in f:
            line = line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print(f"Errore nel parsing della riga in {path.name}")
                continue


def _find_data_file(folderPath: Path, suffix: str) -> Path | None:
    """Trova il file .jsonl il cui nome termina con il suffisso indicato (es. '_posts', '_comments')."""
    matches = sorted(Path(folderPath).glob(f"*{suffix}.jsonl"))
    return matches[0] if matches else None


def _subreddit_folders(rawDataFolder: str) -> list[Path]:
    return sorted(entry for entry in Path(rawDataFolder).iterdir() if entry.is_dir())


def _processed_csv_path(jsonlFile: Path) -> str:
    """Percorso del csv processato: stessa cartella e stesso nome del .jsonl, con suffisso '_processed'."""
    return str(jsonlFile.with_name(jsonlFile.stem + "_processed.csv"))


def _save_dataframe(df: pd.DataFrame, path: str, fields: list[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, columns=fields, index=False, encoding="utf-8")


#  STEP 1 - Processamento del dump dei post di un subreddit

def processPostDumps(subredditFolder: str) -> pd.DataFrame:
    """
    Step 1: legge il dump .jsonl dei post di UN singolo subreddit
    (subredditFolder) e lo raccoglie in un DataFrame, senza applicare alcun filtro.
    """
    subredditFolder = Path(subredditFolder)
    postsFile = _find_data_file(subredditFolder, "_posts")
    if postsFile is None:
        raise FileNotFoundError(f"Nessun file di post trovato in {subredditFolder}")

    print(f"- Processing posts file: {postsFile.name}")
    rows = []
    for post in _read_jsonl(postsFile):
        rows.append({
            "id": post.get("id", ""),
            "author": post.get("author", ""),
            "subreddit": post.get("subreddit", subredditFolder.name),
            "title": post.get("title", "") or "",
            "selftext": post.get("selftext", "") or "",
        })

    df_posts = pd.DataFrame(rows, columns=["id", "author", "subreddit", "title", "selftext"])
    print(f"[STEP 1] Post totali raccolti: {len(df_posts):,}")
    return df_posts


#  STEP 2 - Filtraggio dei post per keywords

def filterPostsByKeywords(df_posts: pd.DataFrame) -> pd.DataFrame:
    """
    Step 2: filtra i post del subreddit tenendo solo quelli che contengono
    almeno una keyword (in titolo o corpo). Aggiunge la colonna 'matched_keywords'.
    """
    combined = (df_posts["title"].fillna("") + " " + df_posts["selftext"].fillna(""))
    is_match = combined.apply(_text_matches)

    df_filtered = df_posts.loc[is_match].copy()
    df_filtered["matched_keywords"] = combined.loc[is_match].apply(_find_keywords)

    print(f"[STEP 2] Post trovati per keyword: {len(df_posts):,} -> {len(df_filtered):,}")

    return df_filtered.reset_index(drop=True)


#  STEP 3 - Lista degli id dei post considerati

def getConsideredPostIds(df_posts_filtered: pd.DataFrame) -> set[str]:
    """
    Step 3: costruisce l'insieme degli id (formato "t3_xxxxx") dei post del
    subreddit che hanno superato il filtro keyword. Verrà usato per
    selezionare i commenti pertinenti nello step 6.
    """
    considered_ids = {f"t3_{post_id}" for post_id in df_posts_filtered["id"]}
    print(f"[STEP 3] Id di post considerati: {len(considered_ids):,}")
    return considered_ids


#  STEP 4 - Pulizia dei post

def cleanPosts(df_posts_filtered: pd.DataFrame, subredditFolder: str) -> pd.DataFrame:
    """
    Step 4: rimuove dai post filtrati per keyword quelli non utili alla ricerca:
    - post con contenuto cancellato/rimosso (selftext o author in DELETED_MARKERS)
    - post pubblicati da bot (es. AutoModerator)
    - post duplicati (stesso id)
    Il risultato viene salvato in un csv accanto al file .jsonl dei post.
    """
    df = df_posts_filtered.copy()
    n_before = len(df)

    is_deleted = df["selftext"].isin(DELETED_MARKERS) | df["author"].isin(DELETED_MARKERS)
    is_bot = df["author"].isin(BOT_AUTHORS)

    df_no_deleted_bot = df.loc[~is_deleted & ~is_bot]
    df_clean = df_no_deleted_bot.drop_duplicates(subset="id")
    n_duplicates = len(df_no_deleted_bot) - len(df_clean)

    print(f"[STEP 4] Pulizia post: {n_before:,} -> {len(df_clean):,} "
          f"(rimossi: {is_deleted.sum():,} cancellati/rimossi, {is_bot.sum():,} bot, "
          f"{n_duplicates:,} duplicati)")

    df_clean = df_clean.reset_index(drop=True)
    savePosts(df_clean, subredditFolder)

    return df_clean


def savePosts(df_posts: pd.DataFrame, subredditFolder: str):
    """Salva i post processati in un csv nella cartella del subreddit, con lo
    stesso nome del file .jsonl dei post e suffisso '_processed'."""
    subredditFolder = Path(subredditFolder)
    postsFile = _find_data_file(subredditFolder, "_posts")
    if postsFile is None:
        raise FileNotFoundError(f"Nessun file di post trovato in {subredditFolder}")

    outPath = _processed_csv_path(postsFile)
    _save_dataframe(df_posts, outPath, POST_FIELDS)
    print(f"[SAVE] {len(df_posts):,} post salvati in {outPath}")


#  STEP 5 - Processamento del dump dei commenti di un subreddit

def processCommentDumps(subredditFolder: str) -> pd.DataFrame:
    """
    Step 5: legge il dump .jsonl dei commenti di UN singolo subreddit
    e lo raccoglie in un DataFrame, senza applicare alcun filtro.
    """
    subredditFolder = Path(subredditFolder)
    commentsFile = _find_data_file(subredditFolder, "_comments")
    if commentsFile is None:
        raise FileNotFoundError(f"Nessun file di commenti trovato in {subredditFolder}")

    print(f"- Processing comments file: {commentsFile.name}")
    rows = []
    for comment in _read_jsonl(commentsFile):
        rows.append({
            "id": comment.get("id", ""),
            "link_id": comment.get("link_id", "") or "",
            "parent_id": comment.get("parent_id", "") or "",
            "subreddit": comment.get("subreddit", subredditFolder.name),
            "author": comment.get("author", ""),
            "body": comment.get("body", "") or "",
        })

    df_comments = pd.DataFrame(
        rows, columns=["id", "link_id", "parent_id", "subreddit", "author", "body"]
    )
    print(f"[STEP 5] Commenti totali raccolti: {len(df_comments):,}")
    return df_comments


#  STEP 6 - Filtraggio dei commenti sui post selezionati

def filterCommentsByPosts(df_comments: pd.DataFrame, considered_post_ids: set[str]) -> pd.DataFrame:
    """
    Step 6: filtra i commenti che sono:
    - di primo livello (risposta diretta al post, parent_id == link_id)
    - relativi a un post che ha superato il filtro keyword (STEP 3)
    """
    is_first_level = df_comments["parent_id"] == df_comments["link_id"]
    is_considered = df_comments["link_id"].isin(considered_post_ids)

    df_filtered = df_comments.loc[is_first_level & is_considered].copy()

    print(f"[STEP 6] Commenti trovati: {len(df_comments):,} -> {len(df_filtered):,}")

    return df_filtered.drop(columns=["parent_id"]).reset_index(drop=True)


#  STEP 7 - Salvataggio dei commenti processati

def saveComments(df_comments_filtered: pd.DataFrame, subredditFolder: str) -> pd.DataFrame:
    """
    Step 7: salva i commenti processati del subreddit in un csv nella stessa
    cartella del file .jsonl dei commenti, con lo stesso nome e suffisso '_processed'.
    """
    subredditFolder = Path(subredditFolder)
    commentsFile = _find_data_file(subredditFolder, "_comments")
    if commentsFile is None:
        raise FileNotFoundError(f"Nessun file di commenti trovato in {subredditFolder}")

    outPath = _processed_csv_path(commentsFile)
    _save_dataframe(df_comments_filtered, outPath, COMMENT_FIELDS)
    print(f"[SAVE] {len(df_comments_filtered):,} commenti salvati in {outPath}")
    return df_comments_filtered


#  STEP 8 - Pulizia ed esplorazione finale, pre stance analysis

def cleanAndExploreFinalData(
    df_posts: pd.DataFrame, df_comments: pd.DataFrame, subredditFolder: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Step 8: pulizia ed esplorazione finale di post e commenti del subreddit
    prima della stance analysis, per garantire che i dati siano completi,
    coerenti e rappresentativi del fenomeno studiato.

    Pulizia dei commenti:
      1. rimozione dei commenti privi di link_id (non associabili a un post)
      2. rimozione dei commenti con body nullo, vuoto o in DELETED_MARKERS
      3. rimozione dei commenti con author nullo o "[deleted]"
      4. rimozione dei commenti pubblicati da bot (es. AutoModerator)
      5. rimozione di eventuali duplicati (stesso id)

    Pulizia dei post:
      - rimozione di eventuali duplicati (stesso id), residuo di dump sovrapposti

    Esplorazione:
      - conteggio finale di post/commenti
      - lunghezza media dei testi

    Il risultato viene salvato in due csv nella cartella del subreddit,
    accanto ai rispettivi file .jsonl (stesso nome + '_processed').
    """
    # --- Pulizia commenti ---
    df_c = df_comments.copy()
    n_before = len(df_c)

    has_link_id = df_c["link_id"].notna() & (df_c["link_id"] != "")
    body = df_c["body"]
    has_body = body.notna() & (body.str.strip() != "") & (~body.isin(DELETED_MARKERS))
    author = df_c["author"]
    has_author = author.notna() & (author != "[deleted]")
    is_not_bot = ~author.isin(BOT_AUTHORS)

    df_comments_clean = df_c.loc[has_link_id & has_body & has_author & is_not_bot]
    df_comments_clean = df_comments_clean.drop_duplicates(subset="id").reset_index(drop=True)

    print(f"[STEP 8] Pulizia commenti: {n_before:,} -> {len(df_comments_clean):,}")

    # --- Pulizia post ---
    df_posts_clean = df_posts.drop_duplicates(subset="id").reset_index(drop=True)
    print(f"[STEP 8] Pulizia post: {len(df_posts):,} -> {len(df_posts_clean):,}")

    # --- Esplorazione ---
    avg_post_len = (df_posts_clean["title"].str.len() + df_posts_clean["selftext"].str.len()).mean()
    avg_comment_len = df_comments_clean["body"].str.len().mean()
    print(f"[STEP 8] Post finali: {len(df_posts_clean):,}, lunghezza media testo: {avg_post_len:.1f} caratteri")
    print(f"[STEP 8] Commenti finali: {len(df_comments_clean):,}, lunghezza media testo: {avg_comment_len:.1f} caratteri")

    savePosts(df_posts_clean, subredditFolder)
    saveComments(df_comments_clean, subredditFolder)

    return df_posts_clean, df_comments_clean


#  ESECUZIONE COMPLETA DELLA PIPELINE PER UN SINGOLO SUBREDDIT

def runSubredditPipeline(subredditFolder: str):
    """Esegue in sequenza tutti gli step della pipeline per un singolo subreddit
    (equivalente a eseguirli uno a uno nel notebook)."""
    df_posts_raw = processPostDumps(subredditFolder)
    df_posts_filtered = filterPostsByKeywords(df_posts_raw)
    considered_post_ids = getConsideredPostIds(df_posts_filtered)
    df_posts_clean = cleanPosts(df_posts_filtered, subredditFolder)

    df_comments_raw = processCommentDumps(subredditFolder)
    df_comments_filtered = filterCommentsByPosts(df_comments_raw, considered_post_ids)
    saveComments(df_comments_filtered, subredditFolder)

    df_posts_final, df_comments_final = cleanAndExploreFinalData(
        df_posts_clean, df_comments_filtered, subredditFolder
    )

    print("\nDone :>")
    return df_posts_final, df_comments_final


if __name__ == "__main__":
    for folder in _subreddit_folders(RAW_DATA_FOLDER):
        print(f"\n===== {folder.name} =====")
        runSubredditPipeline(str(folder))
