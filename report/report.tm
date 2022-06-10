<TeXmacs|2.1.1>

<style|generic>

<\body>
  <doc-data|<\doc-title>
    Project Report

    <paragraph|Machine Learning for Comuter Vision>
  </doc-title>|<doc-author|<author-data|<author-name|Hanying
  Zhang>|<\author-affiliation>
    University of Bologna
  </author-affiliation>|<\author-affiliation>
    <date|>

    \;
  </author-affiliation>>>>

  \;

  \;

  <section|Executive Summary>

  In this project the Vision Transformer(ViT)<\footnote>
    \;
  </footnote> paper is studied and implemented. An experiment is also carried
  out to test the performace of the model. The visual attention and
  positional embedding are also studied.

  <section|Vision Transformer>

  Transformer<\footnote>
    \;
  </footnote> was firstly introduced in Natural Language Processing field and
  has dominated the field since then. Inspired by the success of Transformer
  in NLP, the Vision Transformer(ViT) paper applied a standard transformer
  directly to images, using as few as possible modifications. The paper shows
  that when trained on large(14M-300M) images, ViT could get excellent
  results when pre-trained at sufficient scale and transferred to tasks with
  fewer datapoints.

  <subsection|Model Overview>

  \;

  <subsection|Model Components>

  \;

  <section|Experiments>

  The main purpose of this experiment is using a new dataset to fine tune the
  ViT model. After finishing the fine tuning, the positinal embedding and
  attenion are to be shown.

  \;

  <subsection|Dataset>

  The dataset here is Dogs Vs. Cats. However, the competetionn is already
  finished and thus the test is not available. There's still another website
  who still accepts test submissions. But the dataset is different from the
  one on Kaggle. The training data contains 20k images and the validation and
  test sets both contain 2k images.

  The Dataset and DataLoader classes are used to customized a new Dataset
  class to import all this dataset.

  <subsection|Fine Tune>

  The performance before fine tuning is also important as a benchmark. After
  loading the weights into the model, the head needs to be modified according
  to the number of classes in the new dataset. As the new dataset has only
  two classes, the new head is s FC layer whose weight is [2, 768], where 768
  is the output dim of the transformer. During the first experiment, all
  parameters are fixed except for the ones in the new head.

  Then, during the fine tuning process, all parameters are changable.

  In both experiments, the training process run for 5 epochs, the batch size
  are both 16 and the optimizer are both Adam. However, the learning rate in
  the first experiment is 0.0003, in the second experiment is 0.0001. \ 

  <subsection|Results and Analysis>

  The accurate before and after fine tune are 75.2% and 89.3%.

  It's obvious that after fine tuning, the accuracy would become better.

  <subsection|Positinal Embedding>

  \;

  <subsection|Visual Attention>

  <section|Conclusion>

  In this project an experiment was carried out to test the performance of
  ViT as backbone in a classification task. Specifically, the new dataset was
  used to test the model before and after fine tuning. This experiment proved
  that ViT has the ability ...\ 

  After fine tuning, the positional embedding and vision Attention were also
  studied. According to the figures observed, it is convincable that the 1-D
  positional embedding and visual attention are both functioning very well.

  More experiments, such as self supervised learning and other types of
  positional embeddings, could also be carried out on this model to unveil
  more information of ViT model.

  \;
</body>

<\initial>
  <\collection>
    <associate|page-medium|paper>
  </collection>
</initial>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|1>>
    <associate|auto-10|<tuple|3.4|?>>
    <associate|auto-11|<tuple|3.5|?>>
    <associate|auto-12|<tuple|4|?>>
    <associate|auto-2|<tuple|1|?>>
    <associate|auto-3|<tuple|2|?>>
    <associate|auto-4|<tuple|2.1|?>>
    <associate|auto-5|<tuple|2.2|?>>
    <associate|auto-6|<tuple|3|?>>
    <associate|auto-7|<tuple|3.1|?>>
    <associate|auto-8|<tuple|3.2|?>>
    <associate|auto-9|<tuple|3.3|?>>
    <associate|footnote-1|<tuple|1|?>>
    <associate|footnote-2|<tuple|2|?>>
    <associate|footnr-1|<tuple|1|?>>
    <associate|footnr-2|<tuple|2|?>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|1<space|2spc>Vision
      Transformer> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>
    </associate>
  </collection>
</auxiliary>