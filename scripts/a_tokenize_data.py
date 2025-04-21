import os
from utils.tokenizer import CharTokenizer
from utils.constants import (RAW_DATA_PATH, 
                             TOKENIZER_PATH,
                             PROCESSED_DATA_PATH,
                             PROCESSED_PATH)


def tokenize_data():
    with open(RAW_DATA_PATH, 'r') as f:
        text = f.read()

    tokenizer = CharTokenizer(text)

    os.makedirs(PROCESSED_PATH, exist_ok=True)
    tokenizer.save(TOKENIZER_PATH)

    tokens = tokenizer.encode(text)
    tokens_s = ','.join(str(token) for token in tokens)

    
    with open(PROCESSED_DATA_PATH, 'w') as f:
        f.write(tokens_s)

    print("Tokens saved as a string to", PROCESSED_DATA_PATH)

    return tokenizer, tokens

if __name__ == "__main__":
    tokenize_data()