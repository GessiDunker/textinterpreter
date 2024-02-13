import os
import nltk

from flask import Flask, render_template
from tag_definitions import get_full_tagged_counts

nltk.download('brown')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk import RegexpParser
from nltk.corpus import brown
from collections import Counter


def get_pos_tag_counts(file_name):
    text = open(file_name, "r").read()
    tokens = nltk.word_tokenize(text.lower())
    text = nltk.Text(tokens)
    tags = nltk.pos_tag(text)
    pos_tags = [tag for _, tag in tags]
    tag_counts = Counter(pos_tags)
    total_words = len(tokens)
    return get_full_tagged_counts(tag_counts), total_words

app = Flask(__name__, template_folder='web')

full_tagged_human, total_words_human = get_pos_tag_counts("text-human.txt")
full_tagged_gpt, total_words_gpt = get_pos_tag_counts("text-gpt2.txt")

all_keys = set(full_tagged_human.keys()) | set(full_tagged_gpt.keys())
table_tag_counts = []
for key in all_keys:
    count_human = full_tagged_human.get(key, 0)
    count_gpt = full_tagged_gpt.get(key, 0)
    table_tag_counts.append((key, count_human, count_gpt))

table_tag_counts.sort(key=lambda x: x[0])

@app.route("/")
def index():
    return render_template('index.html', total_gpt=total_words_gpt, total_human=total_words_human, combined_tag_counts=table_tag_counts)    

if __name__ == "__main__":
    print("----- App Start -----")
    app.run(port=int(os.environ.get('PORT', 81)))
