import os
import re
import glob
import argparse
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import PyPDF2
import nltk
import nltk.corpus
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud


# Must be performed once to get English nltk corpus:
# nltk.download()


def filter_terms(raw_list: List[str]) -> List[str]:
    """
    Process list of raw index terms (remove page numbers, extract abbrevations,
    lemmatize) and combine them in unique list of index terms.
    Args:
        raw_list: List of index terms in unprocessed form

    Returns:
        List of unique, processed index terms
    """

    def extract_ml_terms(s: str) -> List[str]:
        """ Extract referenced term term1 see term2 -> [term1, term2] """
        # Extract pattern: "see <term>"
        match = re.findall(r'see.*?[A-Z]', s)
        if match:
            first_term = match[0][4:-1]
            second_term = s[len(first_term) + 4:]
            return [first_term, second_term]

        return [s]

    # Strip referenced page numbers.
    index_lst = np.hstack([re.split(r', [0-9]*', c) for c in raw_list])
    index_lst = [re.sub(r' ([0-9]+[-–]*[0-9]*)', '', c) for c in index_lst]
    index_lst = [c for c in index_lst if not (c == '' or c == '\n')]
    index_lst = np.hstack([extract_ml_terms(c) for c in index_lst])
    # Remove page numbers.
    index_lst = [c for c in index_lst if
                 not bool(re.match(r'(.*?[0-9]*\n)', c))]
    index_lst = [c for c in index_lst if not bool(re.match(r'(.*?,[0-9])', c))]
    index_lst = [c for c in index_lst if not c.isdigit()]

    # Extract abbreviations.
    # E.g., artificial neural network (ann) -> [artificial neural network, ann]
    index_lst = np.hstack([extract_abbreviation(c) for c in index_lst])

    index_lst = [c.lower() for c in index_lst]

    # Split index words consisting of multiple words before lemmatization.
    wnl = WordNetLemmatizer()
    index_lst = [' '.join([wnl.lemmatize(x) for x in w.split()])
                 for w in index_lst]

    return np.unique(index_lst)


def extract_abbreviation(s: str) -> List[str]:
    """
    Extract abbreviations, return long term and abbreviation.
    E.g., artificial neural network (ann) -> [artificial neural network, ann]
    Args:
        s: string term with abbreviation in brackets

    Returns:
        [long term, abbreviation]
    """
    match = re.findall(r'\(.*?\)', s)
    if match:
        long = s[:s.find('(')].strip()
        abbrv = re.sub(r'\(|\)', '', match[0])
        return [long, abbrv]
    return [s]


def get_text_from_pdf(filepath: str) -> str:
    """
    Open academic paper PDF file, extract text from pages, remove references,
    and process text (lemmatize, remove stop words, small caps).
    Args:
        filepath: absolute file path to PDF file

    Returns:
        Processed text extracted from file path
    """
    wnl = WordNetLemmatizer()
    stop_words = set(nltk.corpus.stopwords.words('english'))
    pdfReader = PyPDF2.PdfFileReader(open(filepath, 'rb'))
    text = ''
    for i in range(pdfReader.getNumPages()):
        text += pdfReader.getPage(i).extractText()

    # Remove references.
    try:
        pos_ref = re.search(r'\n(References|REFERENCES)\n', text).span()[0]
        text = text[:pos_ref]
    except AttributeError:
        print(f'Removing references failed for {filepath}')

    # Tokenize and lemmatize each word in text.
    text = text.replace('\n', '')
    text = word_tokenize(text)
    text = [wnl.lemmatize(w) for w in text]
    text = [w for w in text if w not in stop_words]
    text = ' '.join(text)

    return text.lower()


def write_unique_index_terms(filepath_out: str, filepaths_in,
                             encoding: str = 'utf-8'):
    """
    Combine list of index term files (e.g., from multiple books or proceedings)
    into single list of unique index terms and write them to file.
    This file path can be adapted manually to fit the specific needs.
    Args:
        filepath_out: absolute file path of file with joined index terms
        filepaths_in: list of absolute file path of file to be joined
        encoding: file encoding

    Returns:

    """
    contents = []
    for filepath in filepaths_in:
        with open(filepath, 'r', encoding=encoding) as file:
            contents += file.readlines()
    index_lst = filter_terms(contents)

    print('Number of extracted ML/AI index terms:', len(index_lst))

    index_lst = list(filter_terms(contents))

    with open(filepath_out, 'w', encoding='utf-8') as outfile:
        outfile.write('\n'.join(index_lst))


def create_word_cloud(text: str, index_lst: List[str],
                      filepath: str = 'wordcloud.png',
                      stem: bool = True,
                      figsize: Tuple[int, int] = (16, 9),
                      **args) -> Dict[str, int]:
    """
    Create word cloud from text given a list of index terms. Stem text and index
    terms for higher matching rate.
    Args:
        text: joined text from all documents/papers
        index_lst: list of index terms
        filepath: file path to word cloud output
        stem: True if index terms and text is to be stemmed
        figsize: word cloud figure size
        **args: arguments for customization of word cloud

    Returns:
        Dict containing counts of index terms in text.
    """
    ps = PorterStemmer()

    def stem_term(term: str) -> str:
        """ Stem term, which can consist of multiple words. """
        return ' '.join([ps.stem(i) for i in term.split()])

    tokenized = text.split()
    if stem:
        tokenized = [ps.stem(i) for i in tokenized]
        text = ' '.join(tokenized)

    # Count index terms in text.
    dct = {}
    for i in index_lst:
        if stem:
            term = stem_term(i)
        else:
            term = i

        if i.count(' ') > 0:
            # Index term consists of multiple words: use str.count()
            dct[i] = text.count(term)
        else:
            # Single word: count in tokenized text.
            dct[i] = tokenized.count(term)

    if stem:
        # Unify stemmed keys.
        stemmed_keys = [stem_term(i) for i in dct.keys()]
        dct_wordcloud = {}
        for stemmed_key in stemmed_keys:
            terms = [i for i in index_lst if stem_term(i) == stemmed_key]
            dct_wordcloud[terms[0]] = sum([dct[i] for i in terms])
    else:
        dct_wordcloud = dct

    # Generate and plot word cloud.
    wc = WordCloud(background_color="white", **args)
    wc.generate_from_frequencies(dct_wordcloud)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(filepath)
    plt.show()

    return {k: v for k, v in dct_wordcloud.items() if v > 0}


def str2bool(x: str) -> bool:
    return x.lower() == 'true'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Machine learning terms word cloud visualization.')
    add_arg = parser.add_argument
    add_arg('--filepath_index', type=str, default='index-terms.txt', help='File path to index term txt file. Default: index-terms.txt')
    add_arg('--filepath_word_cloud', type=str, default='wordcloud.png', help='File path to index term txt file. Default: wordcloud.png')
    add_arg('--stem', type=str2bool, default=True, help='Apply word stemming to text and index terms. Default: True.')
    add_arg('--pdfs', nargs="+", type=str, help='List of PDF files (or single folder) as input for word cloud generation. If folder is specified, use all PDF files in this folder.')
    add_arg('--figsize', type=tuple, default=(16, 9), help='Figure/Plot size. Controls resolution. Default: (16, 9).')
    add_arg('--relative_scaling', type=float, default=.2, help='Importance of relative word frequencies for font-size in word cloud (between 0 and 1). Default: 0.5.')
    add_arg('--max_font_size', type=int, default=72, help='Maximum font size in word cloud plot. Default: 100.')
    add_arg('--scale', type=int, default=60, help='Controls coarseness of fit for the words. Default: 20.')
    add_arg('--height', type=int, default=400, help='Word cloud canvas height. Default: 400.')
    add_arg('--width', type=int, default=600, help='Word cloud canvas width. Default: 600.')

    args = parser.parse_args()

    # Create joined text file from specified PDF files.
    if len(args.pdfs) == 1 and os.path.isdir(args.pdfs[0]):
        filepaths_pdf = glob.glob(os.path.join(args.pdfs[0], '*.pdf'))
    else:
        filepaths_pdf = args.pdfs
        assert all([os.path.splitext(fp)[1] == '.pdf'
                    for fp in filepaths_pdf]), 'Specify PDF files only!'

    print(f'Reading {len(filepaths_pdf)} PDF files...')
    texts = [get_text_from_pdf(fp) for fp in filepaths_pdf]
    joined_text = ' '.join(texts)

    # Read index term file.
    with open(args.filepath_index, 'r', encoding='utf-8') as infile:
        index_terms = infile.read().split('\n')

    # Create word cloud and save to file. Word cloud parameters are set
    # heuristically for pleasant visualization.
    print('Creating word cloud...')
    dct_out = create_word_cloud(joined_text, index_terms,
                                stem=args.stem,
                                figsize=args.figsize,
                                relative_scaling=args.relative_scaling,
                                max_font_size=args.max_font_size,
                                scale=args.scale,
                                height=args.height,
                                width=args.width)
