from myfuncs import load_data_en
from myfuncs import print_running_time
from myfuncs import get_resource_path
from stanfordcorenlp import StanfordCoreNLP
import platform 


model_dir = '/Users/karloar/Documents/other/ltp_data_v3.4.0'
if platform.system() == 'Windows':
    model_dir = r'E:\ltp_data'
train_file = get_resource_path("data/TRAIN_FILE.TXT")
test_file = get_resource_path("data/TEST_FILE_FULL.TXT")


def get_avg_length(sents: list, nlp: StanfordCoreNLP) -> float:
    x = []
    for sent in sents:
        word_list = nlp.word_tokenize(sent)
        x.append(len(word_list))
    return sum(x) / len(x)


@print_running_time
def main():
    train_sents, _ = load_data_en(train_file)
    test_sents, _ = load_data_en(test_file)
    nlp = StanfordCoreNLP('http://localhost', port=9000, lang='en')
    train_avg_length = get_avg_length(train_sents, nlp)
    test_avg_length = get_avg_length(test_sents, nlp)
    print('train sentences average length:', train_avg_length)
    print('test sentences length:', test_avg_length)
    nlp.close()


if __name__ == '__main__':
    main()
