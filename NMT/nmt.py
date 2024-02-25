!pip install numpy tensorflow tensorflow_text

import numpy as np
import tensorflow as tf
import tensorflow_text as tf_text
import pathlib


from collections import Counter

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

path_to_file = pathlib.Path("por-eng/por.txt")#/content/por-eng/por.txt

np.random.seed(1234)
tf.random.set_seed(1234)


def load_data(path):
    text = path.read_text(encoding="utf-8")

    lines = text.splitlines()
    pairs = [line.split("\t") for line in lines]

    context = np.array([context for target, context, _ in pairs])
    target = np.array([target for target, context, _ in pairs])

    return context, target

portuguese_sentences, english_sentences = load_data(path_to_file)
sentences = (portuguese_sentences, english_sentences)

BUFFER_SIZE = len(english_sentences)
BATCH_SIZE = 64
is_train = np.random.uniform(size=(len(portuguese_sentences),)) < 0.8

train_raw = (
    tf.data.Dataset.from_tensor_slices(
        (english_sentences[is_train], portuguese_sentences[is_train])
    )
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
)
val_raw = (
    tf.data.Dataset.from_tensor_slices(
        (english_sentences[~is_train], portuguese_sentences[~is_train])
    )
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
)

"""#### Normalising the Text"""

def tf_lower_and_split_punct(text):
    text = tf_text.normalize_utf8(text, "NFKD")
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, "[^ a-z.?!,¿]", "")
    text = tf.strings.regex_replace(text, "[.?!,¿]", r" \0 ")
    text = tf.strings.strip(text)
    text = tf.strings.join(["[SOS]", text, "[EOS]"], separator=" ")
    return text

max_vocab_size = 12000


english_vectorizer = tf.keras.layers.TextVectorization(standardize=tf_lower_and_split_punct, max_tokens=max_vocab_size, ragged=True)
english_vectorizer.adapt(train_raw.map(lambda context, target: context))

portuguese_vectorizer = tf.keras.layers.TextVectorization(standardize=tf_lower_and_split_punct, max_tokens=max_vocab_size, ragged=True)
portuguese_vectorizer.adapt(train_raw.map(lambda context, target: target))

"""
#### Text Processor wrapper"""

def process_text(context, target):
    context = english_vectorizer(context).to_tensor()
    target = portuguese_vectorizer(target)
    targ_in = target[:, :-1].to_tensor()
    targ_out = target[:, 1:].to_tensor()
    return (context, targ_in), targ_out



train_data = train_raw.map(process_text, tf.data.AUTOTUNE)
val_data = val_raw.map(process_text, tf.data.AUTOTUNE)

print(f"English (to translate) sentence:\n\n{english_sentences[-5]}\n")
print(f"Portuguese (translation) sentence:\n\n{portuguese_sentences[-5]}")

print(f"First 10 words of the english vocabulary:\n\n{english_vectorizer.get_vocabulary()[:10]}\n")
print(f"First 10 words of the portuguese vocabulary:\n\n{portuguese_vectorizer.get_vocabulary()[:10]}")

vocab_size = portuguese_vectorizer.vocabulary_size()
word_to_id = tf.keras.layers.StringLookup(
    vocabulary=portuguese_vectorizer.get_vocabulary(),
    mask_token="",
    oov_token="[UNK]"
)


id_to_word = tf.keras.layers.StringLookup(
    vocabulary=portuguese_vectorizer.get_vocabulary(),
    mask_token="",
    oov_token="[UNK]",
    invert=True,
)

unk_id = word_to_id("[UNK]")
sos_id = word_to_id("[SOS]")
eos_id = word_to_id("[EOS]")
baunilha_id = word_to_id("baunilha")

for (to_translate, sr_translation), translation in train_data.take(1):
    print(f"Tokenized english sentence:\n{to_translate[0, :].numpy()}\n\n")
    print(f"Tokenized portuguese sentence (shifted to the right):\n{sr_translation[0, :].numpy()}\n\n")
    print(f"Tokenized portuguese sentence:\n{translation[0, :].numpy()}\n\n")


VOCAB_SIZE = 12000
UNITS = 256



class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, units):
        super(Encoder, self).__init__()

        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=units,
            mask_zero=True
        )

        self.rnn = tf.keras.layers.Bidirectional(
            merge_mode="sum",
            layer=tf.keras.layers.LSTM(
                units=units,
                return_sequences=True
            ),
        )


    def call(self, context):

        x = self.embedding(context)
        x = self.rnn(x)

        return x

#testing if any breaks
encoder = Encoder(VOCAB_SIZE, UNITS)



class CrossAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.mha = (
            tf.keras.layers.MultiHeadAttention(
                key_dim=units,
                num_heads=1
            )
        )

        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

    def call(self, context, target):
        attn_output = self.mha(
            query=target,
            value=context
        )

        x = self.add([target, attn_output])
        x = self.layernorm(x)

        return x

#testing if any breaks
attention_layer = CrossAttention(UNITS)

class Decoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, units):
        super(Decoder, self).__init__()
        self.embedding =tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=units,
            mask_zero=True
        )
        # The RNN before attention
        self.pre_attention_rnn = tf.keras.layers.LSTM(
            units=units,
            return_sequences=True,
            return_state=True
        )
        self.attention = CrossAttention(units=units)

        # The RNN after attention
        self.post_attention_rnn = tf.keras.layers.LSTM(
            units=units,
            return_sequences=True
        )

        # The dense layer with logsoftmax activation
        self.output_layer = tf.keras.layers.Dense(
            units=vocab_size,
            activation=tf.nn.log_softmax
        )

    def call(self, context, target, state=None, return_state=False):
        x = self.embedding(target)
        x, hidden_state, cell_state = self.pre_attention_rnn(x, initial_state=state)

        x = self.attention(context, x)
        x = self.post_attention_rnn(x)

        # Compute the logits
        logits = self.output_layer(x)

        if return_state:
            return logits, [hidden_state, cell_state]

        return logits

decoder = Decoder(VOCAB_SIZE, UNITS)


class Translator(tf.keras.Model):
    def __init__(self, vocab_size, units):

        super().__init__()
        self.encoder = Encoder(vocab_size=vocab_size, units=units)
        self.decoder = Decoder(vocab_size=vocab_size, units=units)
    def call(self, inputs):
        context, target = inputs
        encoded_context= self.encoder(context)

        logits = self.decoder(encoded_context,target)
        return logits

translator = Translator(VOCAB_SIZE, UNITS)
logits = translator((to_translate, sr_translation))

def masked_loss(y_true, y_pred):

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    loss = loss_fn(y_true, y_pred)

    # Check which elements of y_true are padding
    mask = tf.cast(y_true != 0, loss.dtype)

    loss *= mask
    # Return the total.
    return tf.reduce_sum(loss)/tf.reduce_sum(mask)


def masked_acc(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, y_true.dtype)
    match = tf.cast(y_true == y_pred, tf.float32)
    mask = tf.cast(y_true != 0, tf.float32)

    return tf.reduce_sum(match)/tf.reduce_sum(mask)


def tokens_to_text(tokens, id_to_word):
    words = id_to_word(tokens)
    result = tf.strings.reduce_join(words, axis=-1, separator=" ")
    return result

def compile_and_train(model, epochs=20, steps_per_epoch=500):
    model.compile(optimizer="adam", loss=masked_loss, metrics=[masked_acc, masked_loss])

    history = model.fit(
        train_data.repeat(),
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_data,
        validation_steps=50,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)],
    )

    return model, history

trained_translator, history = compile_and_train(translator)

def generate_next_token(decoder, context, next_token, done, state, temperature=0.0):

    logits, state = decoder(context, next_token, state=state, return_state=True)
    logits = logits[:, -1, :]

    # If temp is 0 then next_token is the argmax of logits
    if temperature == 0.0:
        next_token = tf.argmax(logits, axis=-1)

    # If temp is not 0 then next_token is sampled out of logits
    else:
        logits = logits / temperature
        next_token = tf.random.categorical(logits, num_samples=1)

    logits = tf.squeeze(logits)
    next_token = tf.squeeze(next_token)
    logit = logits[next_token].numpy()

    # Reshape to (1,1) since this is the expected shape for text encoded as TF tensors
    next_token = tf.reshape(next_token, shape=(1,1))

    # If next_token is End-of-Sentence token you are done
    if next_token == eos_id:
        done = True

    return next_token, logit, state, done

eng_sentence = "I love languages"
texts = tf.convert_to_tensor(eng_sentence)[tf.newaxis]
# Vectorize it and pass it through the encoder
context = english_vectorizer(texts).to_tensor()
context = encoder(context)

next_token = tf.fill((1,1), sos_id)

state = [tf.random.uniform((1, UNITS)), tf.random.uniform((1, UNITS))]
done = False

next_token, logit, state, done = generate_next_token(decoder, context, next_token, done, state, temperature=0.5)
print(f"Next token: {next_token}\nLogit: {logit:.4f}\nDone? {done}")


def translate(model, text, max_length=50, temperature=0.0):
    tokens, logits = [], []
    text = tf.constant([text])
    # Vectorize the text using the correct vectorizer
    context = english_vectorizer(text).to_tensor()
    context = model.encoder(context)
    next_token = tf.constant([[2]])

    state = [tf.zeros((1,UNITS)), tf.zeros((1,UNITS))]

    done = False

    # Iterate for max_length iterations
    for _ in range(max_length):
        next_token, logit, state, done =generate_next_token(decoder, context, next_token, done, state, temperature=0.5)
        if done:
            break

    tokens.append(next_token)
    logits.append(logit)
    tokens = tf.concat(tokens, axis=-1)

    translation = tf.squeeze(tokens_to_text(tokens, id_to_word))
    translation = translation.numpy().decode()

    return translation, logits[-1], tokens

temp = 0.0
original_sentence = "I love languages"

translation, logit, tokens = translate(trained_translator, original_sentence, temperature=temp)

print(f"Temperature: {temp}\n\nOriginal sentence: {original_sentence}\nTranslation: {translation}\nTranslation tokens:{tokens}\nLogit: {logit:.3f}")

def generate_samples(model, text, n_samples=4, temperature=0.6):

    samples, log_probs = [], []
    for _ in range(n_samples):

        _, logp, sample = translate(model, text, temperature=temperature)
        samples.append(np.squeeze(sample.numpy()).tolist())

        # Save the logits
        log_probs.append(logp)

    return samples, log_probs

samples, log_probs = generate_samples(trained_translator, 'I love languages')

for s, l in zip(samples, log_probs):
    print(f"Translated tensor: {s} has logit: {l:.3f}")

def jaccard_similarity(candidate, reference):
    candidate_set = set(candidate)
    reference_set = set(reference)
    common_tokens = candidate_set.intersection(reference_set)
    all_tokens = candidate_set.union(reference_set)
    overlap = len(common_tokens) / len(all_tokens)

    return overlap


def rouge1_similarity(candidate, reference):
    candidate = str(candidate)
    reference = str(reference)
    candidate_word_counts = Counter(candidate)
    reference_word_counts = Counter(reference)

    overlap = 0
    for token in candidate_word_counts.keys():
        token_count_candidate = candidate_word_counts[token]
        token_count_reference = reference_word_counts[token]

        overlap += min(token_count_candidate, token_count_reference)
    precision = overlap / len(candidate)

    recall = overlap / len(reference)

    if precision + recall != 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score



    return 0


def average_overlap(samples, similarity_fn):
    scores = {}
    for index_candidate, candidate in enumerate(samples):

        overlap = 0

        for index_sample, sample in enumerate(samples):
            if index_candidate == index_sample:
                continue
            sample_overlap = similarity_fn(candidate, sample)


            overlap += sample_overlap

        score = overlap / (len(samples) - 1)
        score = round(score, 3)
        scores[index_candidate] = score

    return scores

def weighted_avg_overlap(samples, log_probs, similarity_fn):
    scores = {}
    for index_candidate, candidate in enumerate(samples):
        overlap, weight_sum = 0.0, 0.0
        for index_sample, (sample, logp) in enumerate(zip(samples, log_probs)):
            if index_candidate == index_sample:
                continue

            sample_p = float(np.exp(logp))
            weight_sum += sample_p
            sample_overlap = similarity_fn(candidate, sample)
            overlap += sample_p * sample_overlap

        score = overlap / weight_sum
        score = round(score, 3)
        scores[index_candidate] = score

    return scores

def mbr_decode(model, text, n_samples=5, temperature=0.6, similarity_fn=rouge1_similarity):
    samples, log_probs = generate_samples(model, text, n_samples=n_samples, temperature=temperature)
    scores = weighted_avg_overlap(samples, log_probs, similarity_fn)
    decoded_translations = [tokens_to_text(s, id_to_word).numpy().decode('utf-8') for s in samples]
    max_score_key = max(scores, key=lambda k: scores[k])
    translation = decoded_translations[max_score_key]

    return translation, decoded_translations

english_sentence = "I love languages"

translation, candidates = mbr_decode(trained_translator, english_sentence, n_samples=10, temperature=0.6)

print("Translation candidates:")
for c in candidates:
    print(c)

print(f"\nSelected translation: {translation}")