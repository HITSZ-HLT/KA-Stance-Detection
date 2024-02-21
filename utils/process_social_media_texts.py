import preprocessor as p 
import json
import re
import wordninja
import re
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Data Cleaning
def split_hash_tag(strings):
    
    p.set_options(p.OPT.URL,p.OPT.EMOJI,p.OPT.RESERVED)
    clean_data = p.clean(strings)  # using lib to clean URL, emoji...
    clean_data = clean_data.split(' ')
    
    for i in range(len(clean_data)):
        if clean_data[i].startswith("#") or clean_data[i].startswith("@"):
            clean_data[i] = wordninja.split(clean_data[i]) # split compound hashtags
        else:
            clean_data[i] = [clean_data[i]]
    clean_data = [j for i in clean_data for j in i]

    return ' '.join(clean_data)

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def clean_text(text):
    """Function to clean text using RegEx operations, removal of stopwords, and lemmatization."""
    text_lst = text.split(' ')
    text_lst = [token.lower() for token in text_lst]
    text_lst = [token for token in text_lst if token not in stop_words]
    text_lst = [re.sub(r'[^\w\s]', '', token, re.UNICODE) for token in text_lst]
    text_lst = [token for token in text_lst if token not in stop_words]
    text_lst = [lemmatizer.lemmatize(tag[0], get_wordnet_pos(tag[1])) for tag in pos_tag(text_lst) if get_wordnet_pos(tag[1]) is not None]
    text_lst = [token for token in text_lst if token not in stop_words]
    text = ' '.join(text_lst)
    text = text.lstrip().rstrip()
    return text

def load_slang():
    with open("dataset/replacement_dict/noslang_data.json", "r") as f:
        data1 = json.load(f)
    data2 = {}
    with open("dataset/replacement_dict/emnlp_dict.txt","r") as f:
        lines = f.readlines()
        for line in lines:
            row = line.split('\t')
            data2[row[0]] = row[1].rstrip()
    normalization_dict = {**data1,**data2}

    return normalization_dict

def clean_slang(text):
    norm_dict = load_slang()
    clean_data = text.split(' ')
    
    for i in range(len(clean_data)):
        if clean_data[i].lower() in norm_dict.keys():
            clean_data[i] = norm_dict[clean_data[i].lower()]

    return ' '.join(clean_data)


if __name__ == '__main__':
    string = r'''We still need Elizabeth Warren because of her integrity and competence when it comes to getting things done, especially during the ongoing pandemic caused by coronavirus. She has specific plans for different issues, and we need her to get us through this difficult time. So, let's support Elizabeth Warren's coronavirus plan! #ElizabethWarren #WarrenCoronavirusPlan #ShesGotAPlanForThat #WeNeedWarren #NeverthelessShePersisted #ItsTimeForWomenToLead #WarrenBiden'''
    clean_string = split_hash_tag(string)
    print(string)
    print(clean_string)
    string = r'''*4u and *67'''
    print(string)
    print(clean_slang(string))