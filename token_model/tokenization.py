# encoding=utf-8
import unicodedata


class BasicTokenizer(object):
    def __init__(self, do_lower_case=True, never_split=('[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]')):
        self.do_lower_case = do_lower_case
        self.never_split = never_split

    def _is_control(self, char):  # 判断字符是否是控制符,如果是就去掉（返回True）
        if unicodedata.category(char).startswith('C') and char not in ['\t', '\n', '\r']:
            return True
        return False

    def _is_whitespace(self, char):  # 判断字符是否是代表空格的字符，如果是增加空格
        if char in ['\t', '\r', '\n', ' '] or unicodedata.category(char) == 'Zs':
            return True
        return False

    def _clean_text(self, text):  # 清洗text: 判断是否是代表空格的符号(去掉控制符)，如果是，把符号换成空格
        output = []
        for char in text:
            cp = ord(char)
            if cp in ['0', '0xfffd'] or self._is_control(char):
                continue
            if self._is_whitespace(char):
                output.append(' ')
            else:
                output.append(char)
        return ''.join(output)

    def _is_chinese_char(self, char):  # 判断字符是否是中文
        cp = ord(char)
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or
                (cp >= 0x3400 and cp <= 0x4DBF) or
                (cp >= 0x20000 and cp <= 0x2A6DF) or
                (cp >= 0x2A700 and cp <= 0x2B73F) or
                (cp >= 0x2B740 and cp <= 0x2B81F) or
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or
                (cp >= 0x2F800 and cp <= 0x2FA1F)):
            return True
        return False

    def _tokenize_chinese_chars(self, text):  # 用空格把汉字切分开，其他的符号不处理
        output = []
        for char in text:
            if self._is_chinese_char(char):
                output.append(' ')
                output.append(char)
                output.append(' ')
            else:
                output.append(char)
        return ''.join(output)

    def _run_strip_accents(self, text):  # unicode编码标准化,去掉音调等噪音
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            if unicodedata.category(char) == 'Mn':
                continue
            output.append(char)
        return ''.join(output)

    def _is_punctuation(self, char):  # 判断是否是标点
        cp = ord(char)
        if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64)
                or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
            return True
        if unicodedata.category(char).startswith('p'):
            return True
        return False

    def _run_split_on_punc(self, text):  # 判断是否是标点，如果是，用标点再次切分
        if text in self.never_split:
            return [text]
        output = []
        start_new_words = True
        for char in text:
            if self._is_punctuation(char):
                output.append([char])
                start_new_words = True
            else:
                if start_new_words:
                    output.append([])
                    start_new_words = False
                output[-1].append(char)
        # print('output', output)
        return [''.join(lis) for lis in output]

    def tokenize(self, text):
        text = self._clean_text(text)  # 清洗text: 判断是否是代表空格的符号(去掉控制符)，如果是，把符号换成空格
        text = self._tokenize_chinese_chars(text)  # 用空格把汉字切分开，其他的符号不处理
        # print('text', text)
        orig_tokens = whitespace_tokenize(text)  # 用空格分隔句子
        # print('orig_tokens', orig_tokens)
        # orig_tokens = ['早', '上', '好', '，', '小张,加油']
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = str(token).lower()
                token = self._run_strip_accents(token)  # unicode编码标准化,去掉音调等噪音
                # print('token', token)
            # print('self._run_split_on_punc(token)', self._run_split_on_punc(token))
            split_tokens.extend(self._run_split_on_punc(token))  # 判断是否是标点，如果是，用标点再次切分,对于单字无用；对词语有用
        # print('split_tokens', split_tokens)
        return whitespace_tokenize(' '.join(split_tokens))  # 用空格分隔句子


def whitespace_tokenize(text):  # 用空格分隔句子
    if not str(text).strip():
        return []
    return str(text).strip().split()


def load_vocab(vocab_file):
    vocab = {}
    with open(vocab_file, 'r', encoding='utf-8') as file:
        for word in file.readlines():
            vocab[str(word).strip()] = len(vocab)
    return vocab


class WordPieceTokenizer(object):
    def __init__(self, vocab, max_input_chars_per_word=100, unk_token="[UNK]"):
        self.vocab = vocab
        self.max_input_chars_per_word = max_input_chars_per_word
        self.unk_token = unk_token

    def tokenize(self, text):  # 把word再次切成更细粒度的char,对于单字无用
        output = []
        for token in whitespace_tokenize(text):
            if len(token) > self.max_input_chars_per_word:
                output.append(self.unk_token)
                continue
            start = 0
            is_bad = False
            sub_tokens = []
            while start < len(token):
                end = len(token)
                curr_sub_str = None
                while start < end:
                    sub_str = token[start:end]
                    if start > 0:
                        sub_str = '##' + sub_str
                    if sub_str in self.vocab:
                        curr_sub_str = sub_str
                        break
                    end -= 1
                if curr_sub_str is None:
                    is_bad = True
                    break
                sub_tokens.append(curr_sub_str)
                start = end
            if is_bad:
                output.append(self.unk_token)
            else:
                output.extend(sub_tokens)
        return output


class Tokenizer(object):
    def __init__(self, vocab_file, do_lower_case=True, unk_token="[UNK]"):
        self.un_token = unk_token
        self.token_to_id = load_vocab(vocab_file)
        self.id_to_token = {id: token for token, id in self.token_to_id.items()}
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordPiece_tokenizer = WordPieceTokenizer(self.token_to_id)

    def tokenize(self, text):
        output = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordPiece_tokenizer.tokenize(token):
                output.append(sub_token)
        return output

    def convert_tokens_to_ids(self, tokens):
        return [self.token_to_id.get(token, self.un_token) for token in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.id_to_token.get(id, self.un_token) for id in ids]


# if __name__ == '__main__':
#     text = '你好，张同学。加油'
#     Tokenizer = Tokenizer('vocab.txt')
#     tokens = Tokenizer.tokenize(text)
#     print(tokens)
#     ids = Tokenizer.convert_tokens_to_ids(tokens)
#     print(ids)
#     tokenss = Tokenizer.convert_ids_to_tokens(ids)
#     print(tokenss)





