---
title: How Search Engines Work: Base Search and Inverted Index
parent: Articles
nav_order: 2
---
# How Search Engines Work: Base Search and Inverted Index
<i>[@PashaPodolsky](https://github.com/ppodolsky)</i>

Under the hood of almost every search string beats the same fiery heart - an inverted index. It is the inverted index that takes text queries and returns a list of documents to the user, who looks at all this and enjoys cats, answers from StackOverflow, and wiki pages.

The article describes the structure of the search engine, the inverted index, and its optimizations with references to theory. Tantivy, a Rust implementation of the Lucene architecture, is used as a test subject. The article turned out to be concentrated, mathematical, and incompatible with relaxed reading of the hub over a cup of coffee, beware!

The formal problem setting is: there is a set of text documents, we want to quickly find the most relevant documents in this set based on the search query and add new documents to the set for subsequent search.

As the first step, we will define what document relevance to the query is, and we will do it in a way that is understandable to a computer. As the second step, we will find K the most relevant documents and show them to the user. And then we will make everything work with a nice performance.

## Определение релевантности
"Relevance" in human language means the semantic proximity of a document to a query. In mathematical language, proximity can be expressed through the proximity of vectors. Therefore, for the mathematical expression of relevance, it is necessary to associate vectors in some space from the world of mathematics with documents and queries from the world of people. Then a document will be considered relevant to a query if the document-vector and the query-vector are close in our space. A search model with such a definition of proximity is called a vector search model.

The main problem in the vector search model is the construction of the vector space $V$ and the transformation of documents and queries into $V$. In general, vector spaces and transformations can be any, as long as documents or queries that are close in meaning are mapped to close vectors.

<a href="https://towardsdatascience.com/document-embedding-techniques-fed3e7a6a25d">Modern libraries</a> allow for constructing complex vector spaces with a small number of dimensions and high information content in each dimension with just a few clicks. In this space, all the vector coordinates characterize some aspect of the document or query: theme, mood, length, lexicon, or any combination of these aspects. Often what the vector coordinate characterizes cannot be expressed in human language, but is understood by machines. A simple plan for building such a search is:

- Take your favorite library for building text embeddings, such as <a href="https://fasttext.cc/">fastText</a> or <a href="https://github.com/google-research/bert">BERT</a>, and transform the documents into vectors
- Store the obtained vectors in your favorite K nearest neighbors (k-NN) search library, such as <a href="https://github.com/facebookresearch/faiss">faiss</a>
- Transform the search query into a vector using the same method as for documents
- Find the nearest vectors to the query vector and extract the corresponding documents found in the vectors

A k-NN based search will be very slow if you try to put the entire Internet into it. So, we narrow down the definition of relevance so that everything becomes computationally easier.

<i>Note: Here and onwards "words" in the context of documents and queries will be referred to as "terms" to avoid confusion</i>

Let's represent relevance as two mathematical functions and then fill them with content:

- $inline$score(q, d)$inline$ - the relevance of the document to the query
- $inline$score(t, d)$inline$ - the relevance of the document to one term

We impose the restriction of additivity on $inline$score(q, d)$inline$ and express the relevance of the query through the sum of the relevance of the terms: $$display$$score(q, d)=\sum_{t \in q}score(t, d)$$display$$ Additivity simplifies further computations, but forces us to agree with a strong simplification of reality - as if all words in the text occur <a href="https://en.wikipedia.org/wiki/Bag-of-words_model">independently of each other</a>.

The most well-known additive relevance functions are TF-IDF and BM25. They are used in most search systems as the main relevance metrics.

<h3><font color="#cc0000">The origin of TF-IDF and BM25</font></h3>

If you know how to derive the formulas from the title, you can skip this part.

Both TF-IDF and BM25 measure the relevance of a document to a query with a single number. The higher the value of the metrics, the more relevant the document. The values themselves do not have any significant interpretation. Only the comparison of the values of the functions for different documents is important. One document is more relevant to this query than another if its relevance function value is higher.

Let's try to repeat the reasoning of the authors of the formulas and reproduce the steps of building TF-IDF and BM25. We will denote the size of the corpus of indexed documents as N. The simplest thing to do is to define relevance equal to the number of occurrences of the term (term frequency or tf) in the document: $$score(t, d)=tf(t, d)$$ If we have not one term t, but a query q consisting of several terms, and we want to calculate score(q, d) for this document, what should we do? We remember the constraint of additivity and simply sum up all the separate score(t, d) for the terms from the query: $$score(q, d)=\sum_{t \in q}score(t, d)$$ In the formula above, there is a problem - we do not take into account the different "importance" of different terms. If we have a query "cat and dog", then the most relevant documents will be those that contain 100500 occurrences of the term "and". It is unlikely that this is what the user wants to get.

Fixing the problem by weighing each term according to its importance: $$display$$score(t, d)=\frac{tf(t, d)}{df(t)}$$display$$ where $inline$df(t)$inline$ is the number of documents in the corpus containing term $inline$t$inline$. It turns out that the more frequent a term is, the less important it is and the smaller $inline$score(t, d)$inline$ will be. Terms like "and" will have a huge $inline$df(t)$inline$ and therefore a small $inline$score(t, d)$inline$.

It seems better already, but now there's another problem - the $inline$df(t)$inline$ itself says little. If $inline$df(giraffe) = 100$inline$, and the size of the corpus of indexed texts is 100 documents, then the term "giraffe" is considered very frequent in this case. But if the corpus size is 100 000, then it is already rare.

The dependence of $inline$df(t)$inline$ on $inline$N$inline$ can be eliminated by transforming $inline$df(t)$inline$ into a relative frequency by dividing by $inline$N$inline$: $$display$$score(t, d)=\frac{tf(t, d)}{\frac{df(t)}{N}}=tf(t, d)\frac{N}{df(t)}$$display$$ Now let's assume we have 100 documents, in one of them there is the term "elephant," in two - "giraffe". $inline$\frac{N}{df(t)}$inline$ in the first case will be equal to 100, and in the second - 50. The term "giraffe" will receive two times less points than the term "elephant" just because there are one more document with giraffe than with elephants. We will correct this situation by smoothing the function $inline$\frac{N}{df(t)}$inline$.

Smoothing can be performed in different ways, we will do this by taking the logarithm: $$display$$score(t, d) = tf(t, d)\log\frac{N}{df(t)}$$display$$ We just got TF-IDF. Let's move on to BM25.

It is unlikely that a document containing the term "giraffe" 200 times is twice as good as a document containing the term "giraffe" 100 times. So let's smooth things out again, but now not by logarithm, but a little differently. Replace $inline$tf(t, d)$inline$ with $inline$tf_s(t, d) = \frac{tf(t, d)}{tf(t, d) + k}$inline$. With each increase in the number of term occurrences $inline$tf(t, d)$inline$ by one, the value of $inline$tf_s(t, d)$inline$ grows smaller and smaller - the function is smoothed out. And with the parameter $inline$k$inline$ we can control the curvature of this smoothing. Speaking smarter, the parameter $inline$k$inline$ controls the degree of saturation of the function.

<img src="https://habrastorage.org/webt/gc/qs/sp/gcqssps36boy_gcstlfvcp57ux4.png" align="center" />

<i><font color="#999999">Figure 0: The higher the value of $inline$k$inline$, the more subsequent occurrences of the same term will be taken into account.</font></i>

The function $inline$tf_s(t, d)$inline$ has two remarkable side effects.

Firstly, $inline$score(q, d)$inline$ will be greater for documents that contain all the words in the query than for documents that contain only one word from the query multiple times. The top documents in this case will be more pleasing to the user's eyes and mind, as all the query terms are usually not printed for no reason.

Secondly, the value of the function $inline$tf_s(t, d)$inline$ is upper bounded. The rest of $inline$score(t, d)$inline$ is also upper bounded, so the whole function $inline$score(t, d)$inline$ is upper bounded (further $inline$UB_t$inline$ - upper bound). Moreover, $inline$UB_t$inline$ is very easy to calculate in our case.

Why is $inline$UB_t$inline$ important for us? $inline$UB_t$inline$ is the maximum possible contribution of this term to the relevance function value. If we know $inline$UB_t$inline$, we can prune angles when processing the query.

The final step is to start taking into account the lengths of documents in $inline$score(t, d)$inline$. In long documents, the term "giraffe" may appear simply by chance and its presence in the text tells nothing about the real topic of the document. But if a document consists of one term and this term is "giraffe", then we can confidently assert that the document is about giraffes.

The obvious way to take into account the length of the document is to take the number of words in the document $inline$dl(d)$inline$. Additionally, we will divide $inline$dl(d)$inline$ by the average number of words in all documents $inline$dl_{avg}$inline$. We will do this based on the same considerations as we normalized $inline$df(t)$inline$ above - absolute values degrade the quality of the metric.

Now let's find a place for the document length in our formula. When $inline$k$inline$ grows, $inline$tf_s$inline$ decreases. If we multiply $inline$k$inline$ by $inline$\frac{dl(d)}{dl_{avg}}$inline$, it turns out that longer documents will receive a lower $inline$score(t, d)$inline$. That's what we need!

It is possible to further parameterize the strength with which we consider the length of the document, to control the behavior of the formula in different situations. Let's replace $inline$\frac{dl(d)}{dl_{avg}}$inline$ with $inline$1 - b + b\frac{dl(d)}{dl_{avg}}$inline$ and obtain: $$display$$\frac{tf_s(t, d)}{tf_s(t, d) + k(1 - b + b\frac{dl(d)}{dl_{avg}})}$$display$$ When $inline$b = 0$inline$, the formula degenerates into $inline$\frac{tf_s(t, d)}{tf_s(t, d) + k}$inline$, and when $inline$b = 1$inline$, the formula takes the form $inline$\frac{tf_s(t, d)}{tf_s(t, d) + k\frac{dl(d)}{dl_{avg}}}$inline$.

Once again, $inline$k$inline$ is the strength of the influence of repeating terms on relevance, and $inline$b$inline$ is the strength of the influence of document length on relevance.

Let's substitute $inline$tf$inline$ into $inline$tf_s$inline$:

$$display$$score(q, d)=\sum_{t \in q} \frac{tf(t, d) (k + 1)}{tf(t, d) + k(1 - b + b\frac{dl(d)}{dl_{avg}})} * \log\frac{N}{df(t)}$$display$$

We have obtained the BM25 formula with a minor nuance. In the canonical formula $inline$\log\frac{N}{df(t)}$inline$ (this term is called $inline$IDF$inline$) is replaced by $inline$\log\frac{N - df(t) + 0.5}{df(t) + 0.5}$inline$. This substitution is based on fitting to a theoretically purer form of the <a href="https://t.me/libgen_scihub_bot?start=TklEOiA3NDMwNTc4NQ==">RSJ model</a> and does not have simple heuristics behind it. This form of $inline$IDF$inline$ gives a lower weight to terms that appear too often: articles, conjunctions, and other letter combinations that carry little information.

An important note: from the BM25 formula it is now evident that $inline$UB_t$inline$ is more dependent on the value of $inline$IDF$inline$, that is, on the frequency of the term in the corpus. The rarer the term, the higher its maximum possible contribution to $inline$score(q, d)$inline$.

<h2><font color="#cc0000">Implementation of Inverted Index</font></h2>
Given the limited memory, slow disks, and processors, we now need to devise a data structure capable of producing the top-K BM25 relevant documents.
We have a set of documents for which search is required. All documents are assigned a document ID or DID. Each document is broken down into terms, terms can be truncated or brought to a canonical form if desired. For each processed term, a list of DID documents containing this term is created - a posting list.

<img src="https://habrastorage.org/webt/n1/m2/1u/n1m21uvb8olfafuu5av8kvoaihu.png" alt="terms-and-posting-lists" align="center" />
<i><font color="#999999">Figure 1: Posting lists</font></i>

Different implementations of inverted indexes may also preserve the exact places in the document where a term or the total number of term occurrences in the document occur. This additional information is used in calculating relevance metrics or for executing specific queries where the mutual arrangement of terms in the document is important. The posting list itself is sorted in ascending order of <i>DID</i>, although there are other approaches to its organization.

The second part of the inverted index is a dictionary of all terms. A <a href="https://en.wikipedia.org/wiki/Key%E2%80%93value_database">KV store</a> is used for the dictionary, where terms are the keys and values are the addresses of posting lists in RAM or on disk. Hash tables and trees are usually used for the KV store in memory. However, other structures may be more appropriate for the term dictionary, such as <a href="https://habr.com/ru/post/111874/">prefix trees</a>.

<img src="https://habrastorage.org/webt/wg/am/wz/wgamwzgfeawopd_ei-jtwvgwhnq.png" align="center"/>
<i><font color="#999999">Fig. 2: Term Dictionary (Prefix Tree)</font></i>

In Tantivy, <a href="https://blog.burntsushi.net/transducers/">finite-state transducers</a> are used for term storage through the <a href="https://docs.rs/fst/0.4.5/fst/">fst</a> crate. Simplifying it, prefix trees organize the dictionary by extracting common prefixes of the keys, while transducers can also extract common suffixes. Thus, the compression of the dictionary is performed more efficiently, but in the end, it becomes an acyclic graph instead of a tree.

The fst library can compress even better than general-purpose compression algorithms in extreme cases while still preserving arbitrary access. Extreme cases occur when a large portion of your terms have long common parts. For example, when you store URLs in an inverted index.

The fst library also has serialization and deserialization methods for the dictionary, which greatly simplifies life - storing trees and graphs by hand on disk is still entertainment. Unlike hash tables, fst allows wildcard substitution during key searches. Some people reportedly use the asterisk in search queries, but I haven't seen any.

Используя словарь термов и постинг-листы можно для запроса из одного одного терма $inline$t$inline$ определить список документов, в котором этот терм появляется. Затем останется посчитать $inline$score(t, d)$inline$ для каждого документа из постинг-листа и взять top-K документов. 

Для этого перенесем $inline$score(t, d)$inline$ из области математики в реальный мир. В Tantivy используется BM25, как один из вариантов функции релевантности:
$$display$$score(t, d)=\sum_{t \in q} \frac{tf(t, d) (k + 1)}{tf(t, d) + k(1 - b + b\frac{dl(d)}{dl_{avg}})} * \log\frac{N - df(t) + 0.5}{df(t) + 0.5}$$display$$ 
<ul>
	<li>$inline$tf(t, d)$inline$ - подсчитываем количество <i>DID</i> документа в постинг-листе терма $inline$t$inline$, либо храним отдельным числом, что ускорит весь процесс за счет использования дополнительной памяти</li>
	<li>$inline$df(t)$inline$ - длина всего постинг-листа</li>
        <li>$inline$dl_{avg}$inline$ - рассчитываем на основе двух статистик, общего количества документов в индексе и суммарной длины всех постинг-листов. Обе статистики поддерживаются инвертированным индексом в актуальном состоянии при добавлении нового документа</li>
        <li>$inline$dl(d)$inline$ - храним для каждого документа в отдельном списке</li>
</ul>
Кроме словаря термов, постинг-листов и длин документов Tantivy (и Lucene) сохраняет в файлах:

<ul>
	<li>Точные позиции слов в документах</li>
	<li>Скип-листы для ускорения итерирования по спискам (об этом дальше)</li>
	<li>Прямые индексы, позволяющие извлечь сам документ по его <i>DID</i>. Работают прямые индексы медленно, так как документы хранятся сжатыми тяжелыми кодеками типа <a href="https://github.com/google/brotli">brotli</a></li>
        <li>FastFields - встроенное в инвертированный индекс быстрое KV-хранилище. Его можно использовать для хранения внешних статистик документа а-ля PageRank и использовать их при расчете вашей модифицированной функции $inline$score(q, d)$inline$</li>
</ul>
Теперь, когда мы можем посчитать $inline$score(q, d)$inline$ для запроса из одного терма, найдем top-K документов. Первая идея - посчитать для всех документов их оценки, отсортировать по убыванию и взять К первых. Потребуется хранить всё в RAM и при большом количестве документов у нас кончится память. 

Поэтому при обходе постинг-листа от начала и до конца первые $inline$K$inline$ документов кладутся в <a href="https://en.wikipedia.org/wiki/Heap_(data_structure)">кучу</a> (далее top-K heap) безусловно. А затем каждый последующий документ сначала оценивается и кладется в кучу только если его $inline$score(q, d)$inline$ выше минимального $inline$score(q, d)$inline$ из кучи. Текущий минимум в top-K heap далее будет обозначен как $inline$\theta$inline$.

<h3><font color="#cc0000">Операции над постинг-листами для запросов из нескольких термов</font></h3>
Что сделает инвертированный индекс с запросом "скачать OR котики"? Он заведет два итератора по постинг-листам для термов "скачать" и "котики", начнет итерирование по обоим листам, попутно рассчитывая $inline$score(q, d)$inline$ и поддерживая top-K heap.

Аналогичным образом реализуется AND-запрос, однако тут итерирование позволяет пропускать значительные части постинг-листов без расчета $inline$score(q, d)$inline$ для них.

Более важными для поисковиков общего назначения являются OR-запросы. А всё потому, что они покрывают больше документов и потому, что ранжирование запросов метриками TF-IDF или BM25 всё равно поднимает в топ документы с бОльшим количеством совпавших слов. Это сделает top-K документов больше похожим на результат работы AND-запроса.

Наивный алгоритм реализации OR-запроса следующий:

<ol>
	<li>Создаем итераторы для постинг-листов каждого терма из запроса</li>
	<li>Заводим top-K heap</li>
	<li>Сортируем итераторы (не содержание постинг-листов, а именно набор итераторов) по <i>DID</i>, на которые они указывают в данный момент</li>
	<li>Берем документ, на который указывает первый итератор и собираем среди оставшихся итераторов те, которые указывают на тот же документ. Так мы получим <i>DID</i> и термы, которые содержатся в этом документе</li>
	<li>Рассчитываем релевантность документа по термам, складываем их, получаем релевантность по всему запросу. Если документ хороший, то кладем его в top-K heap</li>
	<li>Продвигаем использованные итераторы и возвращаемся к п.3</li>
</ol>

<img src="https://habrastorage.org/webt/lz/ot/xs/lzotxs7bj4dunbns3tpgd4zxtli.png" align="center"/>
<i><font color="#999999">Рис. 3: Итерации OR-алгоритма. Чуть ниже есть псевдокод алгоритма</font></i>

В п.4 сбор итераторов осуществляется быстро, так как список итераторов отсортирован по <i>DID</i>. Пересортировку итераторов в п.3 тоже можно оптимизировать, если мы знаем какие итераторы были продвинуты в п.6.

<h2><font color="#cc0000">Некоторые оптимизации инвертированного индекса</font></h2>
В обычной задаче поиска ищутся не вообще все релевантные документы, а только K наиболее релевантных. Это открывает путь для важных оптимизаций. Причина простая - большая часть документов станет ненужной и мы избежим накладных вычислений над ней. Такая постановка задачи ещё известна как Top-K queries.

Посмотрим внимательнее на псевдокод OR-алгоритма <a href="https://doi.org/10.1145/3041021.3054191">Bortnikov, 2017</a>:
<source>
Input:
  termsArray - Array of query terms
  k - Number of results to retrieve
Init:
  for t ∈ termsArray do t.init()
  heap.init(k)
  θ ← 0
  Sort(termsArray) 
Loop: 
  while (termsArray[0].doc() < ∞) do
    d ← termsArray[0].doc()
    i ← 1
    while (i < numTerms ∩ termArray[i].doc() = d) do
      i ← i + 1
    score ← Score(d, termsArray[0..i − 1]))
    if (score ≥ θ) then 
      θ ← heap.insert(d, score)
    advanceTerms(termsArray[0..i − 1]) 
    Sort(termsArray)
Output: return heap.toSortedArray()

function advanceTerms(termsArray[0..pTerm]) 
  for (t ∈ termsArray[0..pTerm]) do
    if (t.doc() ≤ termsArray[pTerm].doc()) then 
      t.next()
</source>
Наивный алгоритм работает с асимптотикой $inline$O(LQ\log{Q})$inline$, где L - суммарная длина используемых при обработке запроса постинг-листов, а Q - количество слов в запросе. Немного обнаглев, из оценки можно выкинуть $inline$Q\log{Q}$inline$ - подавляющее большинство пользователей приносит запросы не длиннее какого-то максимума и можно считать $inline$Q\log{Q}$inline$ константой.

На практике, сильнее всего скорость работы инвертированного индекса зависит от размера корпуса (т.е суммарной длины постинг-листов) и частоты запросов. Включенное автодополнение запроса или внутренние аналитические запросы в поисковую систему способны кратно умножить нагрузку на систему. Даже $inline$O(L)$inline$ в такой ситуации оказывается слишком грустной оценкой.

<h3><font color="#cc0000">Сжатие постинг-листов</font></h3>
Размер постинг-листов может достать гигабайтных размеров. Хранить постинг-листы как есть и бегать вдоль них без выдоха - плохая идея. Во-первых, можно не влезть в диск. Во-вторых, чем больше надо читать с диска, тем всё медленнее работает. Поэтому постинг-листы являются первыми кандидатами на сжатие.

Let's recall that a posting-list is an increasing list of DIDs, where DIDs are typically 64-bit unsigned integers. The numbers in the posting-list are not significantly different from each other and lie in a relatively limited range of values from 0 to some number comparable to the number of documents in the corpus.

<b><font color="#cc0000">VarLen Encoding</font></b>

It's strange to waste 8 bytes to encode a small number. So people came up with codes that represent small numbers using a small number of bytes. Such schemes are called variable length encodings. We will be interested in a specific scheme known as <a href="https://developers.google.com/protocol-buffers/docs/encoding#:~:text=Varints%20are%20a%20method%20of,are%20further%20bytes%20to%20come.">varint</a>.

Reading a number in varint is done byte by byte. Each read byte stores a signal bit and 7 bits of payload. The signal bit tells us whether we need to continue reading or if the current byte is the last for this number. The payload is concatenated until the last byte of the number.

Постинг-листы хорошо сжимаются varint'ом в несколько раз, но теперь у нас связаны руки - прыгнуть вперед в постинг-листе через N чисел нельзя, ведь непонятно где границы каждого элемента постинг-листа. Получается, что читать постинг-лист мы можем только последовательно, ни о какой параллельности речи не идет.

<b><font color="#cc0000">SIMD</font></b>

Для возможности параллельного чтения изобрели <a href="https://pisa.readthedocs.io/en/latest/compress_index.html#compression-algorithms">компромиссные схемы</a>, похожие на varint, но не совсем. В таких схемах числа разбиваются на группы по N чисел и каждое число в группе кодируются одинаковым количеством бит, а вся группа предваряется дескриптором, описывающим что в группе находится и как это распаковать. Одинаковая длина запакованных чисел в группе позволяет использовать SIMD-инструкции (SSE3 в Intel) для распаковки групп, что кратно ускоряет время работы. 

Tantivy упаковывает <i>DID</i> в блоки по 128 чисел, а затем пишет блок из 128 частот термов, используя <a href="https://fulmicoton.com/posts/behold-tantivy/">bitpack-кодировку</a>.

<b><font color="#cc0000">Delta-encoding</font></b>

Varint хорошо сжимает малые числа и хуже сжимает большие числа. Так как в постинг-листе находятся возрастающие числа, то с добавлением новых документов качество сжатия будет становиться хуже. Простое изменение - в постинг-листе будем хранить не сами <i>DID</i>, а разницу между соседними <i>DID</i>. Например, вместо [2, 4, 6, 9, 13] мы будем сохранять [2, 2, 2, 3, 4].

Список всё постоянно возрастающих чисел превратится в список небольших неотрицательных чисел. Сжать такой список можно эффективнее, однако теперь для раскодирования i-го числа нам нужно посчитать сумму всех чисел до i-го. Впрочем, это не такая уж и большая проблема, ведь varint и так подразумевает, что чтение списка будет последовательным.

<h3><font color="#cc0000">Skip-lists for iterating through posting lists</font></h3>

As already stated, after compressing posting lists, the array of numbers becomes a linked list. All we can do now is to traverse the list from start to end. Although until now, we haven't needed more, the optimization schemes described in the following sections require the ability to move iterators forward by an arbitrary number of <i>DID</i>s.

There is such a wonderful thing - <a href="https://habr.com/ru/post/139870/">skip-lists</a>. The skip-list lives next to the linked sorted list of numbers and represents a sparse index of the contents of this list. If you want to find X in the list of numbers, the skip-list will explain to you in $inline$O(log(L))$inline$ time where exactly you need to jump to be in your list roughly in the right place before X. After the jump, you already go to X with a regular linear search.

The precision of the jump depends on the amount of memory we can allocate for the skip-list, which is a typical trade-off in algorithms. In Tantivy, movement along the posting-list is implemented using skip-lists. There is a <a href="https://github.com/crossbeam-rs/crossbeam-skiplist">lock-free implementation</a> of a skip-list, but as of the time of writing the article (March 2021), the library does not seem to be well-supported.

In our skip-list implementation, we also need to store partial sums up to the point where we plan to jump. Otherwise, it won't work because we used delta encoding for the posting list.

<h3><font color="#cc0000">Optimizing OR Queries</font></h3>
All optimizations of posting list traversal can be divided into safe and unsafe. As a result of applying safe optimizations, the top-K documents remain unchanged compared to the naive OR algorithm. Unsafe optimizations can give a big speed gain, but they change top-K and may miss some documents.

<b><font color="#cc0000">MaxScore</font></b>

MaxScore is one of the first known attempts to speed up the execution of OR-queries. The optimization is safe and described in <a href="https://doi.org/10.1016/0306-4573%2895%2900020-h">Turtle, 1995</a>.

The essence of the optimization is to split the query terms into two non-intersecting sets: "mandatory" and "optional". Documents that contain terms only from the "optional" set cannot enter the top-K and therefore their posting lists can be skipped forward to the first document that contains at least one "mandatory" term.

Remember the $inline$UB_t$inline$ term introduced in the section on TF-IDF and BM25? I remind you that this is a hypothetical "importance" of the term. Common words have low importance, and specific words have high importance. $inline$UB_t$inline$ is a function of the frequency of the term and can be calculated on the fly based on the size of the posting list.

Having the importance of terms at hand, one can sort all the terms from the query in decreasing order of their importance and calculate the partial sums of importance from the first to the last term. All terms with a partial sum less than the current $\theta$ can be assigned to the "optional" set. A document containing only terms from the "optional" set cannot be rated higher than the sum of the importance of these terms and therefore cannot be rated higher than $\theta$. Such a document will not be included in the final set.

It makes sense to consider only those documents that contain at least one term from the "mandatory" set of terms. Therefore, we can skip the posting lists of "optional" terms until the smallest of the <i>DID</i>s that the "mandatory" term iterators point to. This is where we need skip lists, without them we would have to run through the posting lists sequentially and there would be no gain in speed.

<img src="https://habrastorage.org/webt/n7/q0/ow/n7q0ow79sxorkopt9ieuzj3lbze.png" align="center"/>
<i><font color="#999999">Fig. 4: Scrolling Iterators in MaxScore</font></i>

After each update of the top-K heap, the two sets are rebuilt, and the algorithm terminates when all terms are in the "optional" set.

<b><font color="#cc0000">Weak AND (WAND)</font></b>

WAND также является безопасным методом оптимизации поиска, описанным в <a href="https://doi.org/10.1145/956863.956944">Broder, 2003</a>. В чем-то он похож на MaxScore: также анализирует частичные суммы $inline$UB_t$inline$ и $inline$\theta$inline$.

<ol>
	<li>Все итераторы термов WAND сортирует в порядке <i>DID</i>, на который указывает каждый итератор</li>
	<li>Рассчитываются частичные суммы $inline$UB_t$inline$</li>
	<li>Выбирается <i>pivotTerm</i> - первый терм, чья частичная сумма превосходит $inline$\theta$inline$</li>
	<li>Проверяются все предшествующие <i>pivotTerm</i>'у итераторы. 

Если они указывают на один и тот же документ, то этот документ теоретически может входить в top-K документов и поэтому для него производится полноценный рассчет $inline$score(q, d)$inline$. 

Если хотя бы один из итераторов указывает на <i>DID</i>, меньший чем <i>pivotTerm.DID</i>, то такой итератор продвигается вперед до <i>DID<i>, большего чем</i>pivotTerm.DID</i>.

После этого, мы возвращаемся на первый шаг алгоритма</li>
</ol>

<b><font color="#cc0000">Block-max WAND (BMW)</font></b>

BMW is an extension of the WAND algorithm from the previous section, proposed in <a href="https://doi.org/10.1145/2009916.2010048">Ding, 2011</a>. Instead of using global $inline$UB_t$inline$ for each term, we now split the posting lists into blocks and store $inline$UB$inline$ separately for each block. The algorithm repeats WAND, but also checks the partial sum of $inline$UB$inline$ of the blocks that the iterators are currently pointing to. If this sum is below $inline$\theta$inline$, the blocks are skipped.

Block-level $inline$UB$inline$ estimates of terms are much lower in most cases than global $inline$UB_t$inline$. As a result, many blocks will be skipped and this will save time in calculating the $inline$score(q, d)$inline$ of documents.

To understand the gap between production inverted indexes and academic research, you can delve into the widely known ticket in narrow circles <a href="https://issues.apache.org/jira/browse/LUCENE-4100">LUCENE-4100</a>.

<b><font color="#cc0000">Block Upper Scoring</font></b>

Alongside the $inline$UB$inline$ for a block, other metrics can be stored to help make the decision not to touch this block. Or, adjust the $inline$UB$inline$ itself in the right way so that its value reflects your intent.

The author of the article experimented with a search where only fresh news documents were required. BM25 was replaced with $inline$BM25D(q, d, time) = BM25(q, d) * tp(time)$inline$ where $inline$tp$inline$ is a function that imposes a penalty on outdated documents and takes values from 0 to 1. By changing the formula for $inline$UB$inline$, it was possible to pass 95% of all blocks with outdated news, which significantly speeded up this particular search. The approach of storing block metrics, as well as the calculated and final limit of the relevance function, is conducive to experimentation.

<h2><font color="#cc0000">Preprocessing of the Search Query</font></h2>
Before landing in the inverted index, the query undergoes several stages of processing after entering the search string.

<h3><font color="#cc0000">Разбор поискового запроса</font></h3>
<img src="https://habrastorage.org/webt/r8/9r/ud/r89rudf38p_8rvmbg6u0dsva8do.png" align="center"/>
<i><font color="#999999">Fig. 5: Stages of Search Query Processing</font></i>

First, the query syntax tree is built. Punctuation is discarded from the query, the text is converted to lowercase, tokenized, and stemming, lemmatization, and stop-word elimination may be used. Further, a logical tree of operations is built from the token stream.

Whether tokens are joined by default with the OR or AND operator depends on the index settings. On one hand, connecting through AND can sometimes give a more accurate result, on the other hand, there is a risk of finding nothing at all. Several queries can be compiled, and after their execution, the best option can be chosen based on the size of the result or external metrics.

The logical tree forms the basis of the operation plan. In Tantivy, the corresponding structure is called a Scorer. The Scorer implementation is the center of the inverted index universe because this structure is responsible for iterating over posting lists and for all possible optimizations of this process.

<h3><font color="#cc0000">Query Expansion Stage</font></h3>
Almost always, the user wants to receive not what they are requesting, but something slightly different. And therefore, large search engines have complex systems, the goal of which is to expand your request with additional terms and constructions.

The query is diluted with synonyms, terms are given weights, the history of previous searches is used, filters are added, the search context and a million other hacks are added, which are commercial secrets. This stage of work is called query extension, it is incredibly important for improving search quality.

At the stage of expanding search queries, it is cheap to conduct experiments. Imagine that you want to find out if using morphology in a search will give you any profit. Morphology in this context is the conversion of different word forms to the canonical form (lemma).

Например:
<source lang="cpp">скачивать, скачать, скачал, скачали -> скачать
котики, котиков, котикам -> котик</source>
У вас есть несколько вариантов:

<ul>
	<li>Тяжелый: сразу попытаться научить инвертированный индекс хранить и леммы, и словоформы, а также научиться учитывать их при поиске. Нужно много программировать и проводить переиндексацию корпуса документов. </li>

	<li>Компромиссный: переиндексировать весь корпус документов, приводя все словоформы к леммам на лету, а также лемматизировать приходящие запросы. Меньше программирования, но все так же требуется переиндексация.</li>

	<li>Простой: разбавлять запрос всеми словоформами. В таком случае запрос пользователя "скачка котиков" будет преобразован во что-то типа "(скачка^2 OR скачать OR скачивать OR скачал) AND (котиков^2 OR котики OR котик OR котикам)". Выдача будет выглядеть так, как будто бы мы умеем по-настоящему работать с леммами. Содержимое инвертированного индекса менять не требуется!</li>

</ul>
Все гипотетические затраты процессора на переиндексацию благодаря простому подходу будут перенесены на этап query extension и на обработку такого расширенного запроса. Это сэкономит вам кучу времени разработки. Fail often, fail fast!

<h2><font color="#cc0000">Запись и сегментирование индекса</font></h2>
В архитектуре Lucene инвертированный индекс нарезан на <i>сегменты</i>. Сегмент хранит свою часть документов и <i>является неизменяемым</i>. Для добавления документов мы собираем в RAM новые документы, делаем <i>commit</i> и в этот момент документы из RAM сохраняются в новый сегмент.

Сегменты неизменяемы, потому что часть связанных с сегментом данных (скип-листы или отсортированные постинг-листы) являются неизменяемыми. К ним невозможно быстро добавить данные, так как это потребует перестроения всей структуры данных.

Сегменты могут обрабатывать запросы одновременно, поэтому сегмент является естественной единицей распараллеливания нагрузки. Сложность здесь возникает только в слиянии результатов из разных сегментов, так как нужно выполнять N-Way Merge потоков документов от каждого сегмента.

Тем не менее, много маленьких сегментов - плохо. А такое иногда случается, когда запись ведется небольшими порциями. В таких ситуациях Tantivy запускает процедуру слияния сегментов, превращая много маленьких сегментиков в один большой сегмент. 

При слиянии сегментов часть данных, например сжатые документы, могут быть быстро слиты, а часть - придется перестраивать, что загрузит ваши CPU. Поэтому расписание слияний сильно влияет на общую производительность индекса при постоянной пишущей нагрузке.

<h2><font color="#cc0000">Шардирование</font></h2>
Существует два способа распараллеливания нагрузки на инвертированный индекс: по документам или по термам. 

В первом случае каждый из N серверов хранит только часть документов, но является сам по себе полноценным мини-индексом, во втором - хранит только часть термов для всех документов. Ответы шардов во втором случае требуют дополнительной нетривиальной обработки.
<table>
  <tr>
    <th></th><th>По документам</th><th>По термам</th>
  </tr>
  <tr>
    <th>Нагрузка на сеть</th><td><font color="#006600">Маленькая</font></td><td><font color="#ee0000">Большая</font></td>
  </tr>
  <tr>
    <th>Хранение дополнительных аттрибутов для документа</th><td><font color="#006600">Просто</font></td><td><font color="#ee0000">Сложно</font></td>
  </tr>
  <tr>
    <th>Disk-seek'ов для запроса из K слов на N шардах</th><td><font color="#ee0000">O(K*N)</font></td><td><font color="#006600">O(K)</font></td>
  </tr>
  </tr>
</table> Обычно нагрузка на сеть является большей проблемой, чем работа с диском. Поэтому Google в своих первых индексах использовал разбиение по документам. В Tantivy также удобнее использовать шардирование по документам - сегменты индекса натуральным образом являются шардами и количество приседаний при реализации уменьшается во много раз.

Поскольку в инвертированном индексе перестроение сегментов является сложной операцией, лучше сразу начать использовать схемы типа <a href="https://medium.com/omarelgabrys-blog/consistent-hashing-beyond-the-basics-525304a12ba">Ring</a> или <a href="https://arxiv.org/pdf/1406.2294.pdf">Jump</a> Consistent Hashing для снижения объемов перешардируемых документов при открытии нового шарда.

<h2><font color="#cc0000">Многофазовые поиски и ранжирование</font></h2>
В поисковых системах обычно выделяются две части: базовые поиски и мета-поиск. Базовый поиск ищет по одному корпусу документов. Мета-поиск делает запросы в несколько базовых поисков и хитрым способом сливает результаты каждого базового поиска в единый список. 

Базовые поиски условно можно поделить на одно- и двухфазовые. Выше в статье описан именно однофазовый поиск. Такой поиск ранжирует список документов вычислительно простыми метриками типа BM25, используя кучу различных оптимизаций, и в таком виде отдает пользователю. 

Первая фаза двухфазовых (или даже многофазовых) поисков делает всё тоже самое. А вот на второй фазе происходит переранжирование top-K документов из первой фазы с использованием более тяжелых для вычисления метрик. Такое деление оправдано, поскольку отделяет быструю первую фазу на всем множестве документов от тяжелой второй фазы на ограниченном множестве документов.

На практике часть сложных метрик второй фазы часто со временем прорастает в первую фазу и позволяют сразу корректировать релевантность и обход постинг-листов для повышения качества финального результата.

Кстати, при шардировании индекса удобно сервера второй фазы использовать для агрегирования документов от шардов первой фазы.

<h3><font color="#cc0000">First Ranking Phase</font></h3>

The document-to-query matching metric can be very simple, such as $inline$BM25$, or slightly more complex, such as $inline$BM25 * f(IMP)$, where $inline$IMP$ is the static quality of the document, and $inline$f$ is an arbitrary mapping with a domain of $inline$[0; 1]$.

The restriction on $inline$f$ in this case arises from the optimizations used in the BMW-type index, which do not allow modifying $inline$score(t, d)$ without changing the saved block $inline$UB$.

BM25 is calculated during the work with the posting lists, and additional members like $inline$IMP$ should be stored next to the inverted index. This is the first significant limitation of the first search phase. Through the function $inline$score(q, d)$, in general, too many documents are processed for it to be possible to run for each of them to some external systems for additional document attributes.

In web search, the role of the $inline$IMP$inline$ member is usually performed by PageRank, calculated once every N days on large MapReduce clusters. The calculated PageRank is written to a fast KV-storage of an inverted index, such as FastFields in the case of Tantivy, and is used in the calculation of relevance.

Documents from other sources may have other metrics. For example, for search in scientific articles, it makes sense to use the <a href="https://en.wikipedia.org/wiki/Impact_factor">impact factor</a> or citation index.

<h3><font color="#cc0000">Second phase of ranking</font></h3>
In the second phase of ranking, there is already room to speed up. You have hundreds to thousands of more or less relevant documents obtained from the first phase. The remaining time before the request deadline (usually fractions of a second) can be used to run machine learning, load metrics from external databases, and reorder documents to make the output even more relevant.

Retrieve click information from your statistical database and use user preferences to create a personalized output - have fun as you can. You can even calculate something on the fly, such as a new version of BM25, the formula of which came to your mind after Friday's inspirations. Re-training the ranking formula or the second phase model with the new metric is all that is required.

This stage of work is one of the most engaging in search development and also important for good output quality.

<h2><font color="#cc0000">Search Quality</font></h2>
Search quality control deserves a separate article, here I will just leave a couple of links and give a few practical tips.

The main goal of quality control is testing and monitoring the results of releasing new versions of the index. Any quality metric should be an approximation of user satisfaction to some extent. Keep this in mind when inventing something new.

At the initial stage of search development, any increase in the quality metric usually means a real improvement in search quality. But the closer you are to the upper limit of the theoretically possible quality, the more diverse artifacts will arise when optimizing a certain metric.

You will need a lot of logs. To calculate search quality metrics, it is necessary to store the following information for each user query: the list of <i>DID</i> search results and the values of their relevance function in the current search session, <i>DID</i> and positions of clicked documents, user interaction times with the search, and session identifiers. Weights that characterize the quality of the session can also be stored. This way, it is possible to exclude robot sessions and give greater weights to sessions of assessors (if you have them).

Далее пара метрик, с которых вам стоит начать. Они просто формулируется даже в терминах SQL-запроса, особенно если у вас что-то типа Clickhouse для хранения логов.

<b><font color="#cc0000">Success Rate</font></b>

Самое первое и очевидное - нашёл ли вообще пользователь у вас хоть что-нибудь в рамках сессии. 
<spoiler title="SQL-сниппет">
<source lang="pgsql">
with 1.959964 as z
select
    t,
    round(success_rt + z*z/(2*cnt_sess) - z*sqrt((success_rt*(1 - success_rt) + z*z/(4*cnt_sess))/cnt_sess)/(1 + z*z/cnt_sess), 5) as success_rate__lower,
    round(success_rt, 5) as success_rate,
    round(success_rt + z*z/(2*cnt_sess) + z*sqrt((success_rt*(1 - success_rt) + z*z/(4*cnt_sess))/cnt_sess)/(1 + z*z/cnt_sess), 5) as success_rate__upper
from (
    select
        toDateTime(toDate(min_event_datetime)) as event_datetime,
        $timeSeries as t,
        count(*) as cnt_sess,
        avg(success) as success_rt
    from (
        select
            user_id,
            session_id,
            min(event_datetime) as min_event_datetime,
            max(if($yourConditionForClickEvent, 1, 0)) as success
        from
            $table
        where
            $timeFilter and
            $yourConditionForSearchEvent
        group by
            user_id,
            session_id
    )
    group by
        t,
        event_datetime
)
order by
      t</source>
</spoiler>
<b><font color="#cc0000">MAP@k</font></b>

Хорошее введение в MAP@k, а также в несколько других learning-to-rank метрик <a href="https://habr.com/ru/company/econtenta/blog/303458/">есть на Хабре</a>. Скорее всего первое, что вы посчитаете из серьезных метрик. Метрика характеризует насколько хороший у вас top-K документов, где K обычно берется равным количеству элементов на странице поисковой выдачи.
<spoiler title="SQL-сниппет">
<source lang="pgsql">select
    $timeSeries as t,
    avg(if(AP_10 is null, 0, AP_10)) as MAP_10
from
(
    select
        session_id,
        min(event_datetime) as event_datetime
    from (
        select
            session_id,
            event_datetime
        from
            query_log
        where
            $timeFilter and
            $yourConditionForSearchEvent
    )
    group by
        session_id
) search
left join
(
    select
        session_id,
        sum(if(position_rank.1 <= 10, position_rank.2 / position_rank.1, 0))/10 as AP_10
    from (
        select
            session_id,
            groupArray(toUInt32(position + 1)) as position_array_unsorted,
            arrayDistinct(position_array_unsorted) as position_array_distinct,
            arraySort(position_array_distinct) as position_array,
            arrayEnumerate(position_array) as rank_array,
            arrayZip(position_array, rank_array) as position_rank_array,
            arrayJoin(position_rank_array) as position_rank
        from
            query_log
        where
            $timeFilter and
            $yourConditionForClickEvent
        group by
            session_id
    )
    group by
        session_id
) click
on
    search.session_id = click.session_id
group by
    t
order by
    t</source>
</spoiler>
<h2><font color="#cc0000">Вместо заключения: Google и их первый инвертированный индекс</font></h2>
<img src="https://habrastorage.org/webt/zy/sz/ub/zyszubflm9h-mghecvhpdobtw7y.png" align="center" />
<i><font color="#999999">Рис. 6: Вот что бывает, когда программистов заставляют рисовать схемы против их воли (воспроизведение оригинальной блок-схемы из Brin, 1998)</font></i>

Sergei Brin and Larry Page created the first version of Google and indexed about 24 million documents in 1998. The students implemented document downloading using Python, running a total of 3-4 spider processes. One spider processed about 50 documents per second. The full database was filled in 9 days and downloaded hundreds of GB of data. The index itself was counted in tens of GB.

Google invented its own architecture of an inverted index, different from the one that will be used a year later in the first versions of Lucene. The format of the Google index was as simple as five rubles and, in my opinion, very beautiful.

The foundation of the index is Barrel format files. A Barrel is a text file that stores quadruples ⟨<i>did, tid, offset, info</i>⟩, sorted by ⟨<i>did, offset</i>⟩. In this notation, <i>did</i> is the document ID, <i>tid</i> is the term ID, and <i>offset</i> is the position of the term in the document.

In the original system, there were 64 such Barrel files, each of which was responsible for its own range of terms (<i>tid</i>). A new document received a new <i>did</i> and the corresponding quadruples were added to the end of the Barrel files.

A set of such Barrel files is a direct index that allows you to retrieve a list of terms for a given <i>did</i> with a binary search on <i>did</i> (the files are sorted by <i>did</i>). The inverted index can be obtained from the direct index by reversing the operation - we take and sort all files by ⟨<i>tid, did</i>⟩.

<img src="https://habrastorage.org/webt/ti/go/gt/tigogt8phccamhp0_v8ofk_h0ai.png" align="center"/> 
<i><font color="#999999">Рис. 7: Пересортировка Barrel-файлов</font></i>

Done! Again - we are resorting 64 files simultaneously and obtaining an inverted index from a direct index because now binary search can be used to search by <i>tid</i>.

The format of Barrel files clearly shows the earmarks of MapReduce concepts, fully realized and documented in the work <a href="https://dl.acm.org/doi/10.5555/1251254.1251264">J.Dean, 2004</a>.

There is a lot of tasty material about Google in the public domain. You can start with the original work <a href="https://doi.org/10.1016/s0169-7552%2898%2900110-x">Brin, 1998</a> about the search architecture, then poke around in the <a href="https://web.njit.edu/~alexg/courses/cs345/OLD/F15/solutions/">materials of the New Jersey Institute of Technology</a>, and polish everything off with a <a href="http://research.google.com/people/jeff/WSDM09-keynote.pdf">presentation by J.Dean</a> on the inner workings of the first versions of the index.

Imagine you come to work and they tell you that for the whole next year, you'll need to speed up the code by 20% each month to reconcile the debit with credit. That's how Google lived in the early 2000s. Developers at the company had already played so much that they forcibly relocated index files to external cylinders of hard disk drives - their linear rotation speed was higher and files were read faster.

Fortunately, in 2023 such optimizations are no longer necessary. HDDs are virtually replaced from operational work and the index, starting from the mid-2010s, is fully stored in RAM.

<b><font color="#cc0000">Дополнительная литература:</font></b>

<ul>
	<li><a href="https://doi.org/10.1016/S0169-7552(98)00110-X">The anatomy of a large-scale hypertextual Web search engine</a> Brin, 1998</li>
        <li><a href="https://pisa.readthedocs.io/en/latest/">PISA Project Documentation</a></li>
	<li><a href="https://habr.com/ru/company/yandex/blog/464375/">Как работают поисковые системы</a> @iseg</li>
	<li><a href="https://dl.acm.org/doi/10.1145/2682862.2682870">Compression, SIMD, and Postings Lists</a> Trotman, 2014</li>
        <li><a href="https://doi.org/10.1016/0306-4573%2895%2900020-h">Query evaluation: Strategies and optimizations</a> Turtle, 1995</li>
        <li><a href="https://doi.org/10.1145/956863.956944">Efficient query evaluation using a two-level retrieval process</a> Broder, 2003</li>
        <li><a href="https://doi.org/10.1145/2009916.2010048">Faster top-k document retrieval using block-max indexes</a> Ding, 2011</li>
</ul>