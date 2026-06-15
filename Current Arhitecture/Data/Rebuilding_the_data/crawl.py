import requests   # e "browserul" codului; folosim pentru a trimite cereri HTTP si a primi raspunsul
from bs4 import BeautifulSoup  # instrument de parsing; BS transforma html brut intr un format lizibil pt Python
from urllib.parse import urljoin, urlparse
#urljoin: lipeste fragmentele de link. De ex. daca esti pe google.com si gasesti un link catre /images,
#         il transforma automat in linkul complet: https://google.com/images
#urlparse: exact opusul; descompune un URL in bucatele (protocol, domeniu, cale), ceea ce ajuta la
#          a identifica exact de unde vine un link
import io
import pdfplumber   # pentru a citi si din pdf-uri, nu doar text html
import os
import shutil
from pathlib import Path


visited = set()
MAX_DEPTH = 2
ALLOWED_PREFIXES = [
    "/admitere/",
    "/regulamentul-concursului"
]

# Verifica daca linkul "link" gasit pe o alta pagina apartine aceluiasi site de baza, cel de Admitere UBB,
# sau daca este un link extern care te scoate de pe site.
# Literalmente: descompune ambele linkuri si le compara componenta de domeniu (admitere.ubbcluj.ro)
def is_internal(link, base):
    return urlparse(link).netloc == urlparse(base).netloc

def is_relevant_link(link):
    parsed = urlparse(link)

    path = parsed.path.lower()

    return any(prefix in path for prefix in ALLOWED_PREFIXES)


def extract_text(html):
    soup = BeautifulSoup(html, "html.parser") #transforma textul brut intr-un obiect structurat (un "copac" de elemente)

    main_content = soup.find("div", class_="entry clearfix")

    if not main_content:
        return ""

    # Nu avem nevoie de textul din interiorul tagurilor urmatoare, deci le stergem cu totul
    for tag in main_content(["script", "style", "nav", "footer"]): 
        tag.decompose()

    # get_text ne da "Hello!" in loc de "<p>Hello!</p>"
    return main_content.get_text(separator=" ", strip=True)

def extract_text_from_pdf(content):
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        text = ""
        for page in pdf.pages:
            # Extrage tabelele și le formatează frumos ca text
            table = page.extract_table()
            if table:
                for row in table:
                    text += " | ".join([str(cell) if cell else "" for cell in row]) + "\n"
            # Extrage și restul textului
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text


def extract_links(html, base_url):
    soup = BeautifulSoup(html, "html.parser")
    links = []

    # Gaseste sectiunea principala a paginii pt a evita linkuri irelevante
    main_content = soup.find("div", class_="entry clearfix")

    if not main_content:
        return []

    for a in main_content.find_all("a", href=True): # cauta toate linkurile (eticheta <a>)
        link = urljoin(base_url, a["href"]) # pentru cazul ca e link relativ; daca in pagina scrie doar <a href="contact.html">, il transforma in https://admitere.ubbcluj.ro/contact.html
        
        is_pdf = link.lower().endswith(".pdf")
        
        if is_pdf:  
            banned_words = [
                "subiecte", "barem", "rezultat", "repartizare", 
                "facilitat", "solutionare", "sali", "candidati", "lista"]
            if any(cuvant in link.lower() for cuvant in banned_words):  # evitam pdf-urile cu subiecte sau bareme de concurs
                continue
            if is_internal(link, base_url):
                links.append(link)

        else: # Este HTML
            if is_internal(link, base_url) and is_relevant_link(link):
                links.append(link)

    return links


def crawl(url, depth=0):
    # sa nu accesam prea multe sublinkuri & sa nu le repetam pe aceleasi
    if depth > MAX_DEPTH or url in visited:
        return []

    visited.add(url)

    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/119.0.0.0"} # ca sa nu ne blocheze serverul crezand ca suntem un bot
        response = requests.get(url, headers=headers, timeout=10, stream=True)  # stream=True: descarca intai informatiile despre fisier si decide dupa daca l si descarca pt a i extrage textul

        # --VERIFICARE: Vrem ori HTML, ori pdf
        content_type = response.headers.get('Content-Type', '').lower()

        # Cazul 1: Este pagina HTML
        if 'text/html' in content_type:
            print('Vizitez url: ',url)
            html = response.text        # textul insusi, cu taguri cu tot
            text = extract_text(html)   # textul fara taguri si filtrat
            
            results = []
            if text and len(text.strip()) > 100:
                results.append(text)
            for link in extract_links(html, url):
                results += crawl(link, depth + 1)

            return results
        
        # # Cazul 2: Este pdf
        elif 'application/pdf' in content_type:
            print(f"Extrag din pdf: ", url)
            text = extract_text_from_pdf(response.content)  # folosim .content (bytes), nu text
            return [text]

        # Cazul 3: Orice altceva, ignoram
        else:
            return []
        # --
        
    except Exception as e:
        print(f"Eroare la {url}: {e}")
        return []




urls = [
    "https://www.cs.ubbcluj.ro/admitere/nivel-licenta/admiterea-la-facultatea-de-matematica-si-informatica-nivel-licenta/",
    "https://www.cs.ubbcluj.ro/regulamentul-concursului-mate-info-ubb-2026/",
    "https://www.cs.ubbcluj.ro/admitere/nivel-licenta/calendarul-concursului-de-admitere-nivel-licenta/",
    "https://www.cs.ubbcluj.ro/admitere/nivel-licenta/acte-necesare-la-inscrierea-la-facultate-nivel-licenta/",
    "https://www.cs.ubbcluj.ro/admitere/nivel-licenta/taxe-admitere-facultate-nivel-licenta/",
]


def fetch_all_docs():
    docs = [] 

    for url in urls:
        docs.extend(crawl(url))

    # --- SAVE DOCS & CLEANUP ---

    # 1. Define the folder path
    SCRIPT_DIR = Path(__file__).parent
    output_folder = SCRIPT_DIR / "Markdown_conversion/Crawled_documents"
    
    # 2. CLEAR THE FOLDER: If it already exists, delete it and everything inside
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
        
    # Re-create the empty folder
    os.makedirs(output_folder, exist_ok=True)

    # Deduplicate docs
    unique_docs = list(set(docs))

    # 3. Iterate through the unique documents and save them
    for i, doc_text in enumerate(unique_docs):
        filename = os.path.join(output_folder, f"Document{i+1}.txt")
        
        with open(filename, "w", encoding="utf-8") as file:
            file.write(doc_text)

    return unique_docs
