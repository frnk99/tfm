#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
unpickled_df = pd.read_pickle("dummy_db_2000_TEXT.pkl")
# In[2]:

unpickled_df

# In[3]:
s=unpickled_df[unpickled_df['tematica_principal']=='[Actualitat]']

# In[4]:
s

# In[5]:
count = unpickled_df['tematica_principal'].value_counts()
# In[6]:
count = unpickled_df['tematica_principal'].str.split(', ', expand=True).stack().value_counts()

# In[7]:

count
# In[8]:
unpickled_df=unpickled_df.explode('tematica_principal')
# In[9]:
count = unpickled_df['tematica_principal'].value_counts().index
# In[10]:

for x in count:
    print(x)
# In[11]:
Actualitat_data=unpickled_df[unpickled_df['tematica_principal']== 'Divulgació']
Actualitat_data=Actualitat_data.dropna(axis=0,subset=['permatitol'])[0:20]
Actualitat_entradeta=Actualitat_data['permatitol']


# In[12]:
Actualitat_entradeta

# In[13]:

from transformers import pipeline
from transformers import BertTokenizer
from transformers import RobertaForMaskedLM
from transformers import BertModel
from transformers import BertConfig
from transformers import AutoConfig
from transformers import RobertaTokenizer
from transformers import RobertaConfig
config = AutoConfig.from_pretrained('bert-base-uncased')

# In[14]:

fill_mask = pipeline(
    "feature-extraction",
    model=RobertaForMaskedLM.from_pretrained(r"C:/Users/user/Desktop/PFM/julibert/"),
    tokenizer=RobertaTokenizer.from_pretrained(r"C:/Users/user/Desktop/PFM/julibert/"),
    config=RobertaConfig.from_pretrained(r"C:/Users/user/Desktop/PFM/julibert/config.json")
)
config=RobertaConfig.from_pretrained(r"C:/Users/user/Desktop/PFM/julibert/config.json")
config.output_hidden_states = True
tokenizer = RobertaTokenizer.from_pretrained(r"C:/Users/user/Desktop/PFM/julibert/")
model = RobertaForMaskedLM.from_pretrained(r"C:/Users/user/Desktop/PFM/julibert/", config=config)

# In[57]:

sentence_embeddings=[]
Actualitat_data=unpickled_df
Actualitat_data=Actualitat_data[['entradeta','permatitol']].dropna(axis=0,subset=['entradeta'])
for x in range(0,200):
    print(x)
    #[unpickled_df['tematica_principal']== x]
    Actualitat_data1=Actualitat_data[0:10]
    import torch
    Actualitat_data1['encoded']=Actualitat_data1.apply(lambda row: tokenizer.encode(row['entradeta']), axis=1)
    Actualitat_data1['tensor']=Actualitat_data1.apply(lambda row: torch.LongTensor(row['encoded']), axis=1)
    Actualitat_data1['tensor']=Actualitat_data1.apply(lambda row: torch.LongTensor(row['encoded']), axis=1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Actualitat_data1['tensor2']=Actualitat_data1.apply(lambda row: row['tensor'].to(device), axis=1)
    Actualitat_data1['tensor3']=Actualitat_data1.apply(lambda row: row['tensor2'].unsqueeze(0), axis=1)
    Actualitat_data1['out_model']=Actualitat_data1.apply(lambda row: model(input_ids=row['tensor3']), axis=1)
    Actualitat_data1['hidden_states']=Actualitat_data1.apply(lambda row: row['out_model'][1], axis=1)
    Actualitat_data1['hidden_states2']=Actualitat_data1.apply(lambda row: torch.stack(row['hidden_states'],  dim=0), axis=1)
    Actualitat_data1['token_embeddings']=Actualitat_data1.apply(lambda row: torch.squeeze(row['hidden_states2'], dim=1), axis=1)
    Actualitat_data1['token_embeddings2']=Actualitat_data1.apply(lambda row: row['token_embeddings'].permute(1,0,2), axis=1)
    Actualitat_data1['token_vecs']=Actualitat_data1.apply(lambda row: row['hidden_states2'][-2][0], axis=1)
    Actualitat_data1['sentence_embedding']=Actualitat_data1.apply(lambda row: torch.mean(row['token_vecs'], dim=0), axis=1)
    sentence_embeddings.append(Actualitat_data1['sentence_embedding'])
    Actualitat_data=Actualitat_data.iloc[10:,]

# In[55]:
len(sentence_embeddings)
# In[56]:
sentence_embeddings
# In[262]:

sentence_embeddings=[]
for x in count[0:6]:
    print(x)
    Actualitat_data=unpickled_df[unpickled_df['tematica_principal']== x]
    Actualitat_entradeta=Actualitat_data['permatitol']
    Actualitat_data=Actualitat_data.dropna(axis=0,subset=['permatitol'])[0:20]
    import torch
    Actualitat_data['encoded']=Actualitat_data.apply(lambda row: tokenizer.encode(row['permatitol']), axis=1)
    Actualitat_data['tensor']=Actualitat_data.apply(lambda row: torch.LongTensor(row['encoded']), axis=1)
    Actualitat_data['tensor']=Actualitat_data.apply(lambda row: torch.LongTensor(row['encoded']), axis=1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Actualitat_data['tensor2']=Actualitat_data.apply(lambda row: row['tensor'].to(device), axis=1)
    Actualitat_data['tensor3']=Actualitat_data.apply(lambda row: row['tensor2'].unsqueeze(0), axis=1)
    Actualitat_data['out_model']=Actualitat_data.apply(lambda row: model(input_ids=row['tensor3']), axis=1)
    Actualitat_data['hidden_states']=Actualitat_data.apply(lambda row: row['out_model'][1], axis=1)
    Actualitat_data['hidden_states2']=Actualitat_data.apply(lambda row: torch.stack(row['hidden_states'],  dim=0), axis=1)
    Actualitat_data['token_embeddings']=Actualitat_data.apply(lambda row: torch.squeeze(row['hidden_states2'], dim=1), axis=1)
    Actualitat_data['token_embeddings2']=Actualitat_data.apply(lambda row: row['token_embeddings'].permute(1,0,2), axis=1)
    Actualitat_data['token_vecs']=Actualitat_data.apply(lambda row: row['hidden_states2'][-2][0], axis=1)
    Actualitat_data['sentence_embedding']=Actualitat_data.apply(lambda row: torch.mean(row['token_vecs'], dim=0), axis=1)
    sentence_embeddings.append(Actualitat_data['sentence_embedding'])

# In[34]:

len(sentence_embeddings)

# In[14]:
colors = ['red','blue','green','yellow','orange','purple']

# In[20]:
import numpy as np

# In[21]:

x2=[]

# In[44]:
for x in sentence_embeddings:
    for y in x:
        for z in y:
            x2.append(z.detach().numpy())
# In[47]:
x_entreteniment=np.asarray(x2)
# In[48]:
x_entreteniment.shape
# In[49]:

x_entreteniment

# In[50]:

from sklearn.manifold import TSNE
# In[51]:
X_embedded = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(x_entreteniment)


# In[27]:


import matplotlib.pyplot as plt


# In[28]:


import itertools


# In[24]:


colors = ['red','blue','green','yellow','orange','purple']#,'black','brown','grey'


# In[25]:


colors2=list(itertools.chain.from_iterable(itertools.repeat(x, 20) for x in colors))


# In[29]:


plt.scatter(X_embedded[:, 0], X_embedded[:, 1], marker = "x")#, c=colors2)
plt.show()


# In[119]:


hidden_states = out[1]


# In[120]:


granola_ids = granola_ids.unsqueeze(0)


# In[ ]:


model = model.to(device)
granola_ids = granola_ids.to(device)

model.eval()


# In[73]:


# Convert the string "granola bars" to tokenized vocabulary IDs
#granola_ids = tokenizer.encode('La cua del gat era negre, va veure la Mariona mentre es recollia els cabells fent-se una cua. Sempre hi havia molta cua a la perruqueria, la Teresa sempre duia cua.')
# Print the IDs
print('granola_ids', granola_ids)
print('type of granola_ids', type(granola_ids))
# Convert the IDs to the actual vocabulary item
# Notice how the subword unit (suffix) starts with "##" to indicate 
# that it is part of the previous string
print('granola_tokens', tokenizer.convert_ids_to_tokens(granola_ids))


# In[74]:


import torch


# In[75]:


# Convert the list of IDs to a tensor of IDs 
granola_ids = torch.LongTensor(granola_ids)
# Print the IDs
print('granola_ids', granola_ids)
print('type of granola_ids', type(granola_ids))


# In[76]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = model.to(device)
granola_ids = granola_ids.to(device)

model.eval()


# In[77]:


print(granola_ids.size())
# unsqueeze IDs to get batch size of 1 as added dimension
granola_ids = granola_ids.unsqueeze(0)
print(granola_ids.size())

print(type(granola_ids))
with torch.no_grad():
    out = model(input_ids=granola_ids)


# In[78]:


# the output is a tuple
print((out))
# the tuple contains three elements as explained above)
print(len(out))
# we only want the hidden_states
hidden_states = out[1]
print(len(hidden_states))


# In[79]:


print ("Number of layers:", len(hidden_states), "  (initial embeddings + 12 BERT layers)")
layer_i = 0

print ("Number of batches:", len(hidden_states[layer_i]))
batch_i = 0

print ("Number of tokens:", len(hidden_states[layer_i][batch_i]))
token_i = 0

print ("Number of hidden units:", len(hidden_states[layer_i][batch_i][token_i]))


# In[80]:


out


# In[81]:


sentence_embedding = torch.mean(hidden_states[-1], dim=1).squeeze()
print(sentence_embedding)
print(sentence_embedding.size())


# In[82]:


# `hidden_states` is a Python list.
print('Type of hidden_states: ', type(hidden_states))

# Each layer in the list is a torch tensor.
print('Tensor shape for each layer: ', hidden_states[0].size())


# In[83]:


# Concatenate the tensors for all layers. We use `stack` here to
# create a new dimension in the tensor.
token_embeddings = torch.stack(hidden_states, dim=0)

token_embeddings.size()


# In[84]:


token_embeddings = torch.squeeze(token_embeddings, dim=1)

token_embeddings.size()


# In[85]:


# Swap dimensions 0 and 1.
token_embeddings = token_embeddings.permute(1,0,2)

token_embeddings.size()


# In[86]:


# Stores the token vectors, with shape [22 x 3,072]
token_vecs_cat = []

# `token_embeddings` is a [22 x 12 x 768] tensor.

# For each token in the sentence...
for token in token_embeddings:
    
    # `token` is a [12 x 768] tensor

    # Concatenate the vectors (that is, append them together) from the last 
    # four layers.
    # Each layer vector is 768 values, so `cat_vec` is length 3,072.
    cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
    
    # Use `cat_vec` to represent `token`.
    token_vecs_cat.append(cat_vec)

print ('Shape is: %d x %d' % (len(token_vecs_cat), len(token_vecs_cat[0])))


# In[87]:


# Stores the token vectors, with shape [22 x 768]
token_vecs_sum = []

# `token_embeddings` is a [22 x 12 x 768] tensor.

# For each token in the sentence...
for token in token_embeddings:

    # `token` is a [12 x 768] tensor

    # Sum the vectors from the last four layers.
    sum_vec = torch.sum(token[-4:], dim=0)
    
    # Use `sum_vec` to represent `token`.
    token_vecs_sum.append(sum_vec)

print ('Shape is: %d x %d' % (len(token_vecs_sum), len(token_vecs_sum[0])))


# In[150]:


# `hidden_states` has shape [13 x 1 x 22 x 768]

# `token_vecs` is a tensor with shape [22 x 768]
token_vecs = hidden_states[-2][0]

# Calculate the average of all 22 token vectors.
sentence_embedding = torch.mean(token_vecs, dim=0)


# In[88]:


sentence_embedding


# In[151]:


print('First 5 vector values for each instance of "bank".')
print('')
print("bank vault   ", str(token_vecs_sum[2][:5]))
print("bank robber  ", str(token_vecs_sum[21][:5]))
print("river bank   ", str(token_vecs_sum[27][:5]))
print("river bank   ", str(token_vecs_sum[36][:5]))


# In[155]:


from scipy.spatial.distance import cosine

# Calculate the cosine similarity between the word bank 
# in "bank robber" vs "river bank" (different meanings).
diff_bank = 1 - cosine(token_vecs_sum[36], token_vecs_sum[21])

# Calculate the cosine similarity between the word bank
# in "bank robber" vs "bank vault" (same meaning).
same_bank = 1 - cosine(token_vecs_sum[2], token_vecs_sum[36])

print('Vector similarity for  *similar*  meanings:  %.2f' % same_bank)
print('Vector similarity for *different* meanings:  %.2f' % diff_bank)


# In[ ]:





# In[ ]:





# In[84]:


token_i = 2
layer_i = 2
vec = hidden_states[layer_i][batch_i][token_i]

# Plot the values as a histogram to show their distribution.
plt.figure(figsize=(10,10))
plt.hist(vec, bins=200)
plt.show()


# In[34]:


# get last four layers
last_four_layers = [hidden_states[i] for i in (-1, -2, -3, -4)]
# cast layers to a tuple and concatenate over the last dimension
cat_hidden_states = torch.cat(tuple(last_four_layers), dim=-1)
print(cat_hidden_states.size())

# take the mean of the concatenated vector over the token dimension
cat_sentence_embedding = torch.mean(cat_hidden_states, dim=1).squeeze()
print(cat_sentence_embedding)
print(cat_sentence_embedding.size())


# In[36]:


X1=sentence_embedding.detach().numpy()


# In[37]:


pca = PCA(n_components = 50, random_state = 7)
X1 = pca.fit_transform(X1)


# In[ ]:





# In[8]:


llista=fill_mask('pitjor')


# In[10]:


import numpy as np


# In[11]:


lista1=np.array(llista)


# In[12]:


lista1.shape


# In[13]:


model.extract_features('pitjor')


# In[93]:


wordembs = model.get_input_embeddings()


# In[94]:


wordembs


# In[95]:


model.config.vocab_size


# In[97]:


import torch


# In[46]:


allinds = np.arange(0,model.config.vocab_size,1)
inputinds = torch.LongTensor(allinds)
bertwordembs = wordembs(inputinds).detach().numpy()


# In[ ]:





# In[47]:


inputinds


# In[99]:


bertwordembs.shape


# In[ ]:





# In[100]:


def loadLines(filename):
    print("Loading lines from file", filename)
    f = open(filename,'r')
    lines = np.array([])
    for line in f:
        lines = np.append(lines, line.rstrip())
    print("Done. ", len(lines)," lines loaded!")
    return lines


# In[102]:


bertwords = loadLines('vocab.txt')


# In[103]:


model.extract_features(sentence)


# In[29]:


inp = "frase"


# In[30]:


sentence = tok.encode(inp, padding='max_length', max_length=512, truncation=True, return_tensors='pt')

output = model(sentence)


# In[31]:


output


# In[38]:


features=output[-1][-1]


# In[88]:


X1=features.cpu().detach().numpy()


# In[90]:


X1=X1.reshape(512, 768)


# In[91]:


X1=np.reshape(X1,-1)


# In[92]:


X1.shape


# In[38]:


from sklearn.decomposition import PCA


# In[39]:


pca = PCA(n_components = 50, random_state = 7)
X1 = pca.fit_transform(X1)


# In[72]:


tsne = TSNE(n_components = 2, perplexity = 10, random_state = 6, 
                learning_rate = 1000, n_iter = 1500)


# In[73]:


X1 = tsne.fit_transform(X1)


# In[74]:


X1.shape


# In[ ]:


X1


# In[78]:


import matplotlib.pyplot as plt 


# In[80]:


import seaborn as sns


# In[82]:


plt.figure(figsize = (20,15))
p1 = sns.scatterplot(x = X["x1"], y = X["y1"], hue = X["position"], palette = "coolwarm")


# In[34]:


import plotly
import numpy as np
import plotly.graph_objs as go
from sklearn.manifold import TSNE


# In[35]:


output[-1][-1]


# In[36]:


def display_tsne_scatterplot_3D(model, user_input=None, words=None, label=None, color_map=None, perplexity = 0, learning_rate = 0, iteration = 0, topn=5, sample=10):

    if words == None:
        if sample > 0:
            words = np.random.choice(list(model.vocab.keys()), sample)
        else:
            words = [ word for word in model.vocab ]
    
    word_vectors = np.array([model[w] for w in words])
    
    three_dim = TSNE(n_components = 3, random_state=0, perplexity = perplexity, learning_rate = learning_rate, n_iter = iteration).fit_transform(word_vectors)[:,:3]


    # For 2D, change the three_dim variable into something like two_dim like the following:
    # two_dim = TSNE(n_components = 2, random_state=0, perplexity = perplexity, learning_rate = learning_rate, n_iter = iteration).fit_transform(word_vectors)[:,:2]

    data = []


    count = 0
    for i in range (len(user_input)):

                trace = go.Scatter3d(
                    x = three_dim[count:count+topn,0], 
                    y = three_dim[count:count+topn,1],  
                    z = three_dim[count:count+topn,2],
                    text = words[count:count+topn],
                    name = user_input[i],
                    textposition = "top center",
                    textfont_size = 20,
                    mode = 'markers+text',
                    marker = {
                        'size': 10,
                        'opacity': 0.8,
                        'color': 2
                    }
       
                )
                
                # For 2D, instead of using go.Scatter3d, we need to use go.Scatter and delete the z variable. Also, instead of using
                # variable three_dim, use the variable that we have declared earlier (e.g two_dim)
            
                data.append(trace)
                count = count+topn

    trace_input = go.Scatter3d(
                    x = three_dim[count:,0], 
                    y = three_dim[count:,1],  
                    z = three_dim[count:,2],
                    text = words[count:],
                    name = 'input words',
                    textposition = "top center",
                    textfont_size = 20,
                    mode = 'markers+text',
                    marker = {
                        'size': 10,
                        'opacity': 1,
                        'color': 'black'
                    }
                    )

    # For 2D, instead of using go.Scatter3d, we need to use go.Scatter and delete the z variable.  Also, instead of using
    # variable three_dim, use the variable that we have declared earlier (e.g two_dim)
            
    data.append(trace_input)
    
# Configure the layout

    layout = go.Layout(
        margin = {'l': 0, 'r': 0, 'b': 0, 't': 0},
        showlegend=True,
        legend=dict(
        x=1,
        y=0.5,
        font=dict(
            family="Courier New",
            size=25,
            color="black"
        )),
        font = dict(
            family = " Courier New ",
            size = 15),
        autosize = False,
        width = 1000,
        height = 1000
        )


    plot_figure = go.Figure(data = data, layout = layout)
    plot_figure.show()
    


# In[37]:


display_tsne_scatterplot_3D(model, user_input, similar_word, labels, color_map, 5, 500, 10000)


# In[17]:


len(tokens)/200


# In[21]:


results_julibert=[]
results3=[]
res3=None
lst3=None
words=[] 
def julibert_test():
    list_masked=[]
    from lxml import etree
    tree = etree.parse("1533101277690.xml")
    root = tree.getroot()
    etree.tostring(tree)
    namespaces={'tt': 'http://www.w3.org/ns/ttml'}
    root.findall(".//tt:span", namespaces)
    if root.findall(".//tt:span", namespaces) is not None:
        text2 = [events.text for events in root.findall(".//tt:span", namespaces) if events.text !=None]
    text=' '.join(text2)
    tokens = nltk.word_tokenize(text)
    list_masked2=[]
                
    for iteration in range(0,5):
        print(iteration)
        for i in range(((iteration*2+1)*100)+99-199,((iteration*2+1)*100)+299-199):

            tokens_2= tokens[0:]
            words.append(tokens_2[i])
            tokens_2[i] = "<mask>"
            list_masked.append(tokens_2[((iteration*2+1)*100)+99-199:((iteration*2+1)*100)+299-199])
            #print(list_masked[i])                
            list_masked2.append(TreebankWordDetokenizer().detokenize(list_masked[i]))
            results3.append(fill_mask(list_masked2[i])[0]['token_str'])
    results4 = [e.replace('ÃŃ','\xc3\xad').replace('Ãł','\xc3\xa0') for e in results3]
    global lst3
    lst3 = [e[1:].encode('iso-8859-1',"ignore").decode('utf-8','ignore') if len(e)>1  else e for e in results4]
    global res3
    res3 = sum(x == y for x, y in zip(lst3, words)) 
    results_julibert.append(res3)


# In[22]:


julibert_test()


# In[37]:


results_julibert=[]
results3=[]
res3=None
lst3=None
words=[] 
def CALBERT_test():
    list_masked=[]
    from lxml import etree
    tree = etree.parse("1533101277690.xml")
    root = tree.getroot()
    etree.tostring(tree)
    namespaces={'tt': 'http://www.w3.org/ns/ttml'}
    root.findall(".//tt:span", namespaces)
    if root.findall(".//tt:span", namespaces) is not None:
        text2 = [events.text for events in root.findall(".//tt:span", namespaces) if events.text !=None]
    text=' '.join(text2)
    tokens = nltk.word_tokenize(text)
    list_masked2=[]
                
    for iteration in range(0,5):
        print(iteration)
        for i in range(((iteration*2+1)*100)+99-199,((iteration*2+1)*100)+299-199):

            tokens_2= tokens[0:]
            words.append(tokens_2[i])
            tokens_2[i] = "[MASK]"
            list_masked.append(tokens_2[((iteration*2+1)*100)+99-199:((iteration*2+1)*100)+299-199])
            #print(list_masked[i])                
            list_masked2.append(TreebankWordDetokenizer().detokenize(list_masked[i]))
            results3.append(calbert_fill_mask(list_masked2[i])[0]['token_str'])
    results4 = [e.replace('ÃŃ','\xc3\xad').replace('Ãł','\xc3\xa0') for e in results3]
    global lst3
    lst3 = [e[1:].encode('iso-8859-1',"ignore").decode('utf-8','ignore') if len(e)>1  else e for e in results4]
    global res3
    res3 = sum(x == y for x, y in zip(lst3, words)) 
    results_julibert.append(res3)


# In[38]:


julibert_test_CALBERT()


# In[39]:


results_julibert


# In[40]:


results_julibert=[]
results3=[]
res3=None
lst3=None
words=[] 
def MULTIBERT_test():
    list_masked=[]
    from lxml import etree
    tree = etree.parse("1533101277690.xml")
    root = tree.getroot()
    etree.tostring(tree)
    namespaces={'tt': 'http://www.w3.org/ns/ttml'}
    root.findall(".//tt:span", namespaces)
    if root.findall(".//tt:span", namespaces) is not None:
        text2 = [events.text for events in root.findall(".//tt:span", namespaces) if events.text !=None]
    text=' '.join(text2)
    tokens = nltk.word_tokenize(text)
    list_masked2=[]
                
    for iteration in range(0,5):
        print(iteration)
        for i in range(((iteration*2+1)*100)+99-199,((iteration*2+1)*100)+299-199):

            tokens_2= tokens[0:]
            words.append(tokens_2[i])
            tokens_2[i] = "[MASK]"
            list_masked.append(tokens_2[((iteration*2+1)*100)+99-199:((iteration*2+1)*100)+299-199])
            #print(list_masked[i])                
            list_masked2.append(TreebankWordDetokenizer().detokenize(list_masked[i]))
            results3.append(unmasker(list_masked2[i])[0]['token_str'])
    results4 = [e.replace('ÃŃ','\xc3\xad').replace('Ãł','\xc3\xa0') for e in results3]
    global lst3
    lst3 = [e[1:].encode('iso-8859-1',"ignore").decode('utf-8','ignore') if len(e)>1  else e for e in results4]
    global res3
    res3 = sum(x == y for x, y in zip(lst3, words)) 
    results_julibert.append(res3)


# In[41]:


MULTIBERT_test()


# In[42]:


results_julibert


# In[34]:


from transformers import pipeline

calbert_fill_mask  = pipeline("fill-mask", model="codegram/calbert-base-uncased", tokenizer="codegram/calbert-base-uncased")


# In[14]:


import nltk


# In[15]:


nltk.download('punkt')


# In[30]:


from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("codegram/calbert-base-uncased")
model = AutoModel.from_pretrained("codegram/calbert-base-uncased")


# In[25]:


from transformers import pipeline

unmasker = pipeline('fill-mask', model='bert-base-multilingual-cased')


# In[16]:


tokens = nltk.word_tokenize(text)


# In[95]:


tokens_2= tokens


# In[82]:


tokens_2[199] = "[MASK]"


# In[83]:


tokens_2


# In[84]:


tokens_2


# In[107]:


list_masked=[]
words=[]
for i in range(199,200):
    tokens_2= tokens[0:]
    words.append(tokens_2[i])
    #print(tokens[199:])
    print(tokens_2[0])
    tokens_2[i] = "[MASK]"
    print(tokens_2[0])
    list_masked.append(tokens_2[199:])
    print(list_masked)


# In[104]:


list_masked


# In[139]:


list_masked2=[]
results=[]
results2=[]
for i in range(0,199):
    list_masked2.append(TreebankWordDetokenizer().detokenize(list_masked[i][399:599]))
    results.append(calbert_fill_mask(list_masked2[i])[0]['token_str'])
    results2.append(unmasker(list_masked2[i])[0]['token_str'])


# In[140]:


lst = [e[1:] if len(e)>1  else e for e in results]


# In[141]:


res1 = sum(x == y for x, y in zip(lst, words[0:200])) 
res2 = sum(x == y for x, y in zip(results2, words[0:200])) 


# In[142]:


res1


# In[143]:


res2


# In[17]:


list_masked=[]
words=[]
for iteration in range(0,2):
    print(((iteration*2+1)*100)+99)
    for i in range(((iteration*2+1)*100)+99,((iteration*2+1)*100)+299):
        tokens_2= tokens[0:]
        words.append(tokens_2[i])
        tokens_2[i] = "<mask>"
        list_masked.append(tokens_2[((iteration*2+1)*100)+99:((iteration*2+1)*100)+299])
        print((((iteration*2+1)*100)+99))


# In[19]:


list_masked[1]


# In[ ]:





# list_masked

# In[129]:


list_masked2=[]
results3=[]
for i in range(0,199):
    list_masked2.append(TreebankWordDetokenizer().detokenize(list_masked[i][399:599]))
    results3.append(fill_mask(list_masked2[i])[0]['token_str'])
   


# In[35]:


print(len(tokens))
int(len(tokens)/200)


# In[61]:


tokens_tensor = tokens_tensor.to('cuda')
segments_tensors = segments_tensors.to('cuda')
model.to('cuda')


# In[105]:


julibert_test()


# In[106]:


results3


# In[107]:


sum(x == y for x, y in zip(lst3, words))


# In[110]:


results_julibert


# In[108]:


lst3


# In[130]:


results4 = [e.replace('ÃŃ','\xc3\xad').replace('Ãł','\xc3\xa0') for e in results3]


# In[131]:


results4


# In[132]:


lst3 = [e[1:].encode('iso-8859-1',"ignore").decode('utf-8','ignore') if len(e)>1  else e for e in results4]
lst4 = [e[1:] if len(e)>1  else e for e in results3]


# In[133]:


lst3


# In[134]:


words


# In[135]:


res3 = sum(x == y for x, y in zip(lst3, words[0:200])) 


# In[136]:


res3


# In[20]:


from nltk.tokenize.treebank import TreebankWordDetokenizer


# In[144]:


tokens_2=TreebankWordDetokenizer().detokenize(tokens[:200])


# In[145]:


tokens_2


# In[146]:


tokens_orig


# In[147]:


results = calbert_fill_mask(tokens_2)
results


# In[148]:


results[0]['token_str']


# In[149]:


unmasker(tokens_2)


# In[ ]:




