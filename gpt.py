from unidecode import unidecode
import re
import numpy as np

nom_du_fichier = "poids.txt"

class Stockage:
    def __init__(self, nom_du_fichier):
        self.nom_du_fichier = nom_du_fichier
        self.ligne = 0
        with open(nom_du_fichier, "r") as f:
            try:
                f.read()[0]
                self.save = 1
            except Exception:
                self.save = 0
    def read_save(self, value):
        if self.save == 1:
            with open(nom_du_fichier, "w") as f:
                f.write(value)
        if self.save == 0:
            with open(nom_du_fichier, "r") as f:
                f.read()[self.ligne]
                self.ligne += 1
                
Data = Stockage(nom_du_fichier)

if Data.save == 1:
    Data.read_save([5, 8, 2, 0.01, 2])
    Nx = 5
    dimension = 8
    heads = 2
    lr_rate = 0.01
    innerlayer = 2
else:
    var = Data.read_save([])
    Nx = var[0]
    dimension = var[1]
    heads = var[2]
    lr_rate = var[3]
    innerlayer = var[4]

class Tokenisation:
    def __init__(self):
        self.dictionnaire_token = {}
        self.index = 0
        
    def indexer(self, txt):
        if txt in self.dictionnaire_token.keys():
            return self.dictionnaire_token[txt]
        else:
            self.dictionnaire_token[txt] = self.index
            self.index += 1
            return self.index - 1

    def tokenize(self, texte):
        résultat = re.split(r"([.;,:!? ])", texte)
        résultat = [unidecode(txt.lower()) for txt in résultat if txt and txt != " "]
        résultat = [self.indexer(txt) for txt in résultat]
        return résultat


class EmbeddingLayer:
    def __init__(self, vocab_size, dimension, lr_rate):
        self.lr_rate = lr_rate
        self.liste_token = []
        self.embeddings = np.random.randn(vocab_size, dimension)
              
    def forward(self, liste_token):
        self.liste_token = liste_token
        return np.array([self.embeddings[indice] for indice in self.liste_token])
    
    def backward(self, loss_grad):
        dL_dembeddings = np.zeros_like(self.embeddings)
        for i in range(len(loss_grad)):
            dL_dembeddings[self.liste_token[i]] += loss_grad[i]
        self.embeddings -= dL_dembeddings * self.lr_rate
    

class LayerNorm:
    def __init__(self, dimension, lr_rate):
        self.dimension = dimension
        self.gamma = np.random.randn(dimension)
        self.beta = np.random.randn(dimension)
        self.lr_rate = lr_rate
        self.mean = 1
        self.mu = 1
        self.Var_matrix = 1
        self.sigma = 1
        self.out = 1
        
    def forward(self, matrice):
        self.mean = np.mean(matrice, axis = 1, keepdims = True)
        self.mu = matrice - self.mean
        self.Var_matrix = np.var(matrice, axis = 1, keepdims = True)
        self.sigma = np.sqrt(self.Var_matrix + 1e-5)
        self.out = self.gamma * (self.mu / self.sigma) + self.beta
        return self.out
    
    def backward(self, loss_grad):
        direct_path = loss_grad * (self.gamma / self.sigma)
        indirect_path = loss_grad * (self.gamma / self.dimension) * np.sum(
            (1 / self.sigma) + (self.mu * self.mu / (self.sigma)**3),
            axis = 1, keepdims = True)
        dL_dX =  direct_path - indirect_path
        dL_dGamma = np.sum(loss_grad * (self.mu / self.sigma), axis = 0)
        dL_dBeta = np.sum(loss_grad, axis = 0)
        self.gamma -= dL_dGamma * self.lr_rate
        self.beta -= dL_dBeta * self.lr_rate
        return dL_dX
    
    
class ScaleDotAttention:
    def __init__(self, dimension, heads, lr_rate):
        self.lr_rate = lr_rate
        if dimension % heads:
            print("DimentionHeadDivisionError: dimention is not divisible by the number of heads")
            exit()    
        self.heads = heads
        self.dimentionperhead = dimension//heads
        self.W_QKVHeads = [(np.random.randn(self.dimentionperhead, self.dimentionperhead),
                            np.random.randn(self.dimentionperhead, self.dimentionperhead),
                            np.random.randn(self.dimentionperhead, self.dimentionperhead)) for _ in range(heads)]
        self.W_O = np.random.randn(dimension, dimension)
        self.head_matrix = []
        self.embedded_matrice = []
        self.head_matrix_part = []
        self.Query = []
        self.Key = []
        self.DotQKsqrt = []
        self.matricesoft = []
        self.Value = []
        self.matricehead = []
        self.matriceheadsres = []
        self.matriceconc = []
        self.out = []
    
    def QueryKeyMult(self, head_matrix_part, nb_head):
        self.head_matrix_part.append(head_matrix_part)
        self.Query.append(head_matrix_part @ self.W_QKVHeads[nb_head][0])
        self.Key.append(np.transpose(head_matrix_part @ self.W_QKVHeads[nb_head][1]))
        self.DotQKsqrt.append((self.Query[-1] @ self.Key[-1])/np.sqrt(self.dimentionperhead))
        self.matricesoft.append(np.array([np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum() for x in self.DotQKsqrt[-1]]))
        self.Value.append(head_matrix_part @ self.W_QKVHeads[nb_head][2])
        self.matricehead.append(self.matricesoft[-1] @ self.Value[-1])
        return self.matricehead[-1]
    
    def forward(self, embedded_matrice):
        self.embedded_matrice = embedded_matrice
        self.head_matrix_part = []
        self.Query = []
        self.Key = []
        self.DotQKsqrt = []
        self.matricesoft = []
        self.Value = []
        self.matricehead = []
        self.matriceheadsres = []
        self.matriceconc = []
        self.out = []
        self.head_matrix = np.array_split(self.embedded_matrice, self.heads, axis = 1)
        self.matriceheadsres = [self.QueryKeyMult(self.head_matrix[nb_head], nb_head) for nb_head in range(self.heads)]
        self.matriceconc = np.concatenate(self.matriceheadsres, axis = 1)
        self.out = self.matriceconc @ self.W_O
        return self.out
    
    def backward(self, loss_grad):
        dL_dW_O = np.transpose(self.matriceconc).dot(loss_grad)
        dL_dmatriceconc = loss_grad.dot(np.transpose(self.W_O))
        dL_dmatriceheadsres = np.array_split(dL_dmatriceconc, self.heads, axis = 1)
        dL_dmatricehead = []
        dL_dValue = []
        dL_dmatricesoft = []
        dL_dDotQKsqrt = []
        dL_dKey = []
        dL_dQuery = []
        dL_dW_QKVHeads = []
        dL_dhead_matrix_part = []
        for nb_head in range(self.heads):
            dL_dmatricehead.append(dL_dmatriceheadsres[nb_head])
            dL_dValue.append(np.transpose(self.matricesoft[nb_head]).dot(dL_dmatricehead[-1]))
            dL_dmatricesoft.append(dL_dmatricehead[-1].dot(np.transpose(self.Value[nb_head])))
            dL_dDotQKsqrt.append(self.matricesoft[nb_head] * (dL_dmatricesoft[-1] - np.sum((self.matricesoft[nb_head] * dL_dmatricesoft[-1]), axis = 1, keepdims = True)))
            dL_dKey.append(np.transpose(np.transpose(self.Query[nb_head]).dot(dL_dDotQKsqrt[-1])) / np.sqrt(self.dimentionperhead))
            dL_dQuery.append(dL_dDotQKsqrt[-1].dot(np.transpose(self.Key[nb_head])) / np.sqrt(self.dimentionperhead))
            dL_dW_QKVHeads.append((np.transpose(self.head_matrix_part[nb_head]).dot(dL_dQuery[-1]),
                                   np.transpose(self.head_matrix_part[nb_head]).dot(dL_dKey[-1]),
                                   np.transpose(self.head_matrix_part[nb_head]).dot(dL_dValue[-1])))
            dL_dhead_matrix_part.append(dL_dQuery[-1].dot(np.transpose(self.W_QKVHeads[nb_head][0])) + dL_dKey[-1].dot(np.transpose(self.W_QKVHeads[nb_head][1])) + dL_dValue[-1].dot(np.transpose(self.W_QKVHeads[nb_head][2])))
        dL_dembedded_matrice = np.concatenate(dL_dhead_matrix_part, axis = 1)
        self.W_O -= dL_dW_O * self.lr_rate
        for nb_head in range(self.heads):
            self.W_QKVHeads[nb_head] = (self.W_QKVHeads[nb_head][0] - dL_dW_QKVHeads[nb_head][0] * self.lr_rate, self.W_QKVHeads[nb_head][1] - dL_dW_QKVHeads[nb_head][1] * self.lr_rate, self.W_QKVHeads[nb_head][2] - dL_dW_QKVHeads[nb_head][2] * self.lr_rate)
        return dL_dembedded_matrice
    
   
class NeuralNetwork:
    def __init__(self, dimension, innerlayer, lr_rate):
        self.lr_rate = lr_rate
        self.input_size = dimension
        self.innerlayer = innerlayer
        self.hidden_layers = [dimension * 4] * innerlayer
        self.output_size = dimension
        self.Weight = []
        self.Biases = []
        self.Weight.append(np.random.randn(dimension, self.hidden_layers[0]) * 0.1)
        self.Biases.append(np.zeros((1, self.hidden_layers[0])))
        for i in range(self.innerlayer - 1):
            self.Weight.append(np.random.randn(self.hidden_layers[i], self.hidden_layers[i + 1]) * 0.1)
            self.Biases.append(np.zeros((1, self.hidden_layers[i + 1])))
        self.Weight.append(np.random.randn(self.hidden_layers[-1], dimension) * 0.1)
        self.Biases.append(np.zeros((1, dimension)))
        self.layers = []
            
    def forward(self, inputs):
        self.layers = [inputs]
        self.layers.append(np.dot(self.layers[-1], self.Weight[0]) + self.Biases[0])
        for i in range(self.innerlayer):
            self.layers.append(np.dot(self.layers[-1] * (self.layers[-1] > 0), self.Weight[i + 1]) + self.Biases[i + 1])
        return self.layers[-1]
    
    def backward(self, loss_grad):
        dL_dBiases = []
        dL_dWeight = []
        dL_dX = []
        dL_dBiases.insert(0, np.sum(loss_grad, axis = 0))
        dL_dWeight.insert(0, np.transpose(self.layers[-2] * (self.layers[-2] > 0)).dot(loss_grad))
        dL_dX.insert(0, loss_grad.dot(np.transpose(self.Weight[-1])))
        for i in range(self.innerlayer):
            loss_grad = (self.layers[-i-2] > 0) * loss_grad
            if i != self.innerlayer - 1:
                entree = self.layers[-i-3] * (self.layers[-i-3] > 0)
            else:
                entree = self.layers[-i-3]
            dL_dBiases.insert(0, np.sum(loss_grad, axis = 0))
            dL_dWeight.insert(0, np.transpose(entree).dot(loss_grad))
            dL_dX.insert(0, loss_grad.dot(np.transpose(self.Weight[-i-2])))
        for i in range(len(dL_dWeight)):
            self.Weight[i] -= dL_dWeight[i] * self.lr_rate
            self.Biases[i] -= dL_dBiases[i] * self.lr_rate
        return dL_dX[0]


class Transformeur:
    def __init__(self, dimension, heads, innerlayer, lr_rate):
        self.Layernormalisation_1 = LayerNorm(dimension, lr_rate)
        self.Attention = ScaleDotAttention(dimension, heads, lr_rate)
        self.Layernormalisation_2 = LayerNorm(dimension, lr_rate)
        self.Neuronal = NeuralNetwork(dimension, innerlayer, lr_rate)
        
    def forward(self, vecteur):
        vecteur_LN1 = self.Layernormalisation_1.forward(vecteur)
        vecteur_ATT = self.Attention.forward(vecteur_LN1)
        Combined_Vector_1 = vecteur + vecteur_ATT
        vecteur_LN2 = self.Layernormalisation_2.forward(Combined_Vector_1)
        vecteur_NEU = self.Neuronal.forward(vecteur_LN2)
        Combined_Vector_2 = Combined_Vector_1 + vecteur_NEU
        return Combined_Vector_2
    
    def backward(self, loss_grad):
        dL_dvecteur_LN2 = self.Neuronal.backward(loss_grad)
        dL_dCombined_Vector_1 = self.Layernormalisation_2.backward(dL_dvecteur_LN2) + loss_grad
        dL_dvecteur_LN1 = self.Attention.backward(dL_dCombined_Vector_1)
        dL_dvecteur = self.Layernormalisation_1.backward(dL_dvecteur_LN1) + dL_dCombined_Vector_1
        return dL_dvecteur
    

class Generative_Pretrained_Transformer:
    def __init__(self, Nx, dimension, heads, innerlayer, lr_rate):
        self.TransformeurObject = [Transformeur(dimension, heads, innerlayer, lr_rate) for _ in range(Nx)]
        self.Nx = Nx
        self.dimension = dimension
        self.heads = heads
        self.innerlayer = innerlayer
        self.lr_rate = lr_rate
        self.Token = Tokenisation()
        corpus = ["Le Petit Bacchus malade ou Autoportrait en Bacchus est un tableau exécuté par Michelangelo Merisi dit le Caravage, probablement en 1593 voire en 1594, et conservé à Rome dans la galerie Borghèse. Réalisée au début de sa période romaine alors qu'il est âgé d'une vingtaine d'années, il s'agit de l'une des toutes premières œuvres répertoriées du peintre lombard. <EOS>"]
        for texte in corpus:
            self.Token.tokenize(texte)
        self.vocab_size = len(self.Token.dictionnaire_token)
        self.Embedding = EmbeddingLayer(self.vocab_size, dimension, lr_rate)
        self.Layernormalisation_n = LayerNorm(self.dimension, self.lr_rate)
        self.W_out = np.random.randn(dimension, self.vocab_size) * 0.1
        self.matrice_phrase = []
        self.out = []
        
    def generate(self, sentence):
        vecteur = [self.Embedding.forward(self.Token.tokenize(str(sentence)))]
        for Nx in range(self.Nx):
            vecteur.append(self.TransformeurObject[Nx].forward(vecteur[-1]))
        self.matrice_phrase = self.Layernormalisation_n.forward(vecteur[-1])
        logits = self.matrice_phrase @ self.W_out
        self.out = np.array([np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum() for x in logits])
        probs_matrix = self.out[-1]
        probs_matrix += np.random.randn(self.vocab_size) * 0.05
        mot_choisi = probs_matrix.tolist().index(np.max(probs_matrix))
        return list(self.Token.dictionnaire_token.keys())[list(self.Token.dictionnaire_token.values()).index(mot_choisi)]
    
    def backward(self, y_true):
        out_tronque = self.out[:-1]
        n_mots = out_tronque.shape[0]
        Y = np.zeros_like(out_tronque)
        Y[np.arange(n_mots), y_true] = 1
        dL_dlogits = out_tronque - Y
        dL_dlogits = np.vstack([dL_dlogits_tronque, np.zeros((1, self.vocab_size))])
        dL_dW_out = np.transpose(self.matrice_phrase) @ dL_dlogits
        dL_dx = dL_dlogits @ np.transpose(self.W_out)
        self.W_out -= dL_dW_out * self.lr_rate
        loss_grad = self.Layernormalisation_n.backward(dL_dx)
        for Nx in range(self.Nx):
            loss_grad = self.TransformeurObject[self.Nx - Nx - 1].backward(loss_grad)
        self.Embedding.backward(loss_grad)
        
GPT = Generative_Pretrained_Transformer(Nx, dimension, heads, innerlayer, lr_rate)
phrase = "Le Petit Bacchus."
phrase += " <EOS>"
for i in range(67):
    mot = GPT.generate(phrase)
    if mot == "<eos>":
        break
    else:
        phrase += f" {mot}"
print(phrase)
        
