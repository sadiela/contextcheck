import React, { Component } from 'react';
import Header from '../Homepage/Header/HomeButtonRow';

import sourceList from './Sources.png';
import histogram from './Scores_histagram.png';
import algoDiag from './algorithm_diagram.png';
import npov from './npov_edit_example.PNG';

export default class Deets extends Component {
    render() {
        return(
            <>
                <Header />
                <h1>Technical Details</h1>
                <div className='full-deets-wrapper'>
                    <span className='one-entry'>
                        <h2 className='the-question'>
                            How Does ContextCheck Work?
                        </h2>
                        <span className='the-answer-body'>
                            <span className='answer-list-wrapper'>
                                <div className='title-wrapper'>
                                    <h4 className='answer-list-title'>Metadata</h4>
                                </div>
                                <p className='the-answer-list-paragraph'>We scrape article metadata from the URL that is provided. We use the Newspaper3k[12] Python package to do this quickly and efficiently.</p>
                            </span>
                            <span className='answer-list-wrapper'>
                                <div className='title-wrapper'>
                                    <h4 className='answer-list-title'>Related Articles</h4>
                                </div>
                                <p className='the-answer-list-paragraph'>Using NewsAPI[10], we can search for related stories based on keywords and date of publication of a given article. We then link users to five articles on the same topic across the political spectrum of reporting (according to the Adfontes media bias chart [9]). We return one article in each of the following bias categories: partisan left, skew left, center, skew right, and partisan right.</p>
                            </span>
                            <span className='answer-list-wrapper'>
                                <div className='title-wrapper'>
                                    <h4 className='answer-list-title'>Bias Detection</h4>
                                </div>
                                <p className='the-answer-list-paragraph'>Our bias detection model relies on the assumption that news sources should try to be as objective as possible in their reporting. With this in mind, we seek to detect a particular kind of bias: inappropriate subjectivity. Inappropriate subjectivity occurs when language that should be neutral and fair is skewed by feeling, opinion, or taste (whether consciously or unconsciously). In other words, we are looking for a specific type of language that should not be present in objective news reporting. For more information about <strong>inappropriate subjectivity</strong> and how it is defined in linguistic circles, we recommend the paper “Linguistic Models for Analyzing and Detecting Biased Language” by Recasens et al. [7], which describes many of the specific features our model aims to detect. We base our approach to bias detection on Pryzant et al.’s model from their paper “Automatically Neutralizing Subjective Bias in Text”[8].</p>
                            </span>
                        </span>
                    </span>
                    <span className='one-entry'>
                        <h2 className='the-question'>
                            Our Dataset
                        </h2>
                        <span className='the-answer-body'>
                            <p className='answer-paragraph'>
                                Pryzant et al. open-sourced the Wiki Neutrality Corpus (WNC)[8], which is a parallel corpus of 180,000 biased and neutralized sentence pairs. The sentences were collected from Wikipedia edits designed to ensure texts reflected a neutral point of view. In the sentence pairs, words that are changed or removed during a neutrality edit are labeled with a 1 for biased, and all other words are labeled with a 0 for unbiased.
                            </p>
                        </span>
                        <img style={{width:'50vw', margin:'auto'}} src={npov} alt='npov'/>
                    </span>
                    <span className='one-entry'>
                        <h2 className='the-question'>
                            Our Algorithm
                        </h2>
                        <span className='the-answer-body'>
                            <p className='answer-paragraph'>
                                The bias detection algorithm is a BERT [13] model fine-tuned for the specific task of bias detection. The BERT, or Bidirectional Encoder Representations from Transformers model, is pre-trained as a masked-language model on a vast dataset of books and wikipedia articles totaling over 3 billion words. We fine tune for the bias detection task using the WikiNeutrality corpus.
                            </p>
                            <span className='answer-list-wrapper'>
                                <div className='title-wrapper'>
                                    <h4 className='answer-list-title'>Training</h4>
                                </div>
                                <p className='the-answer-list-paragraph'>The goal of the training phase is to fine-tune the BERT model, which is pre-trained on a masked language task, to the new task of bias detection. We do this by passing labeled sentences into the model and performing gradient descent to update the model parameters. We use cross entropy loss and an Adam optimizer.</p>
                            </span>
                            <span className='answer-list-wrapper'>
                                <div className='title-wrapper'>
                                    <h4 className='answer-list-title'>Testing</h4>
                                </div>
                                <p className='the-answer-list-paragraph'>Articles are passed to our model sentence by sentence. When testing a sentence, each word is first tokenized into an integer with the BERT tokenizer and then the integer vector representing the sentence is fed into the pre-trained BERT model. The BERT model outputs a word embedding vector for each word in the sentence. The embedding vector represents each word’s semantic meaning in the context of the sentence. The embedding vector is passed into a linear layer and the final outputs are numeric probabilities (number between 0 and 1) for each word in the sentence. These probabilities represent how certain the algorithm is that a word is biased.</p>
                            </span>
                            <img style={{width:'50vw', margin:'auto'}} src={algoDiag} alt='score histogram'/>
                        </span>
                    </span>
                    <span className='one-entry'>
                        <h2 className='the-question'>
                            How do we calculate the bias score?
                        </h2>
                        <span className='the-answer-body'>
                            <span className='answer-list-wrapper'>
                                <div className='title-wrapper'>
                                    <h4 className='answer-list-title'>5 Steps</h4>
                                </div>
                                <ul className='answer-list'>
                                    <li className='answer-list-item'>For each sentence, take the maximum bias score of any word in the sentence.</li>
                                    <li className='answer-list-item'>Multiply this score by 10 so you have a number between 0 and 10.</li>
                                    <li className='answer-list-item'>Now you have a score on a scale from 0 to 10 for each sentence in the article.</li>
                                    <li className='answer-list-item'>Average the top 50% of these sentence scores.</li>
                                    <li className='answer-list-item'>This is the overall bias score for the article!</li>
                                </ul>
                            </span>
                            <p className='answer-paragraph'>
                                We chose this metric for overall bias because in testing many sentences, we noticed that even in a very biased sentence, this bias often only manifests itself in one or a couple of the words in the sentence. So, the maximum bias of a word in a sentence is a much better indicator of whether or not the sentence is biased than an average score, which is dragged down by many low-scoring words. We average only the top 50% of the sentence scores for a similar reason; not every sentence in an article needs to be biased for that article to contain significant bias. By omitting the lowest 50% of sentences, we again avoid dragging down the score making articles score much lower than they should.
                            </p>
                            <p className='answer-paragraph'>Once we have calculated the numerical score we put the article into one of four buckets.</p>
                            <span className='answer-list-wrapper'>
                                <div className='title-wrapper'>
                                    <h6 className='answer-list-title'>Contains very little to no biased language (0 - 3.5)</h6>
                                </div>
                                <p className='the-answer-list-paragraph'>Our algorithm found barely any biased language! This reporting seems pretty objective.</p>
                            </span>
                            <span className='answer-list-wrapper'>
                                <div className='title-wrapper'>
                                    <h6 className='answer-list-title'>Contains some biased language (3.5 - 5)</h6>
                                </div>
                                <p className='the-answer-list-paragraph'>Our algorithm found some biased language. This reporting is probably objective, but look out for the highlighted words.</p>
                            </span>
                            <span className='answer-list-wrapper'>
                                <div className='title-wrapper'>
                                    <h6 className='answer-list-title'>Contains fair amount of biased language (5 - 6.5)</h6>
                                </div>
                                <p className='the-answer-list-paragraph'>Our algorithm found a fair amount of biased language. This reporting might not be objective.</p>
                            </span>
                            <span className='answer-list-wrapper'>
                                <div className='title-wrapper'>
                                    <h6 className='answer-list-title'>Contains a large amount of biased language (6.5 - 10)</h6>
                                </div>
                                <p className='the-answer-list-paragraph'>Our algorithm found highly biased words in many sentences. This reporting is probably not objective.</p>
                            </span>
                            <span className='answer-list-wrapper'>
                                <h4 className='title-wrapper'>
                                    Why aren't the buckets the same size?
                                </h4>
                                <p className='the-answer-list-paragraph'>
                                  In general, it can be defined as prejudice in favor of or against one thing, person, or group compared with another, usually in a way considered to be unfair. When we read, watch, or listen to the news, we want to get an unbiased, or objective, view of whatever story is being presented. This is important because how we interpret the news affects how we vote, what we think of people, places, and businesses, and so many more things. When making decisions like these, we should be well informed.
                                </p>
                            </span>
                            <img style={{width:'50vw', margin:'auto'}} src={histogram} alt='score histogram'/>
                        </span>
                    </span>
                    <span className='one-entry'>
                        <h2 className='the-question'>
                            Lexicon Definitions and Selected Examples
                        </h2>
                        <span className='the-answer-body'>
                            <span className='answer-list-wrapper'>
                                <div className='title-wrapper'>
                                    <h6 className='answer-list-title'>Assertives</h6>
                                </div>
                                <p className='the-answer-list-paragraph'>Verbs whose complement clauses assert a proposition; the asserting verb can imply different levels of certainty in the claim.
                                    <p><strong>Pointed out</strong> that it is disproved by... (more certainty)</p>
                                    <p><strong>Said</strong> that it is disproved by... (less certainty)</p>
                                </p>
                            </span>
                            <span className='answer-list-wrapper'>
                                <div className='title-wrapper'>
                                    <h6 className='answer-list-title'>Entailments</h6>
                                </div>
                                <p className='the-answer-list-paragraph'>Directional relations that hold whenever the truth of one word/phrase follows from another (e.g., murder entails kill, since there cannot be a murdering without a killing).</p>
                            </span>
                            <span className='answer-list-wrapper'>
                                <div className='title-wrapper'>
                                    <h6 className='answer-list-title'>Factive verbs</h6>
                                </div>
                                <p className='the-answer-list-paragraph'>Verbs that presuppose the truth of their complement clause.
                                    <p>He <strong>realized</strong> ... (Factive)</p>
                                    <p>His stand was that... (neutral)</p>
                                </p>
                            </span>
                            <span className='answer-list-wrapper'>
                                <div className='title-wrapper'>
                                    <h6 className='answer-list-title'>Hedges</h6>
                                </div>
                                <p className='the-answer-list-paragraph'>Used to reduce one’s commitment to the truth of a proposition.
                                    <p>Will decrease...</p>
                                    <p><strong>May</strong> decrease... (Hedge)</p>
                                </p>
                            </span>
                            <span className='answer-list-wrapper'>
                                <div className='title-wrapper'>
                                    <h6 className='answer-list-title'>Implicative</h6>
                                </div>
                                <p className='the-answer-list-paragraph'>Included under entailments; imply the truth or untruth of their complement.</p>
                            </span>
                            <span className='answer-list-wrapper'>
                                <div className='title-wrapper'>
                                    <h6 className='answer-list-title'>Negatives / Positives</h6>
                                </div>
                                <p className='the-answer-list-paragraph'>Words with distinctly negative or positive emotional connotations.
                                    <p>Positive emotional words: excel, easy, thrive</p>
                                    <p>Negative emotional words: terrible, bulky, absurd</p>
                                </p>
                            </span>
                            <span className='answer-list-wrapper'>
                                <div className='title-wrapper'>
                                    <h6 className='answer-list-title'>NPov</h6>
                                </div>
                                <p className='the-answer-list-paragraph'>This word has been changed/removed as part of a wikipedia neutral point of view edit (see description of our corpus above).</p>
                            </span>
                            <span className='answer-list-wrapper'>
                                <div className='title-wrapper'>
                                    <h6 className='answer-list-title'>Reports</h6>
                                </div>
                                <p className='the-answer-list-paragraph'>A verb belonging to a class of verbs conveying the action of speaking and used with both direct and reported speech.
                                    <p>eg. alert, comment, maintain</p>
                                </p>
                            </span>
                            <span className='answer-list-wrapper'>
                                <div className='title-wrapper'>
                                    <h6 className='answer-list-title'>Strong / Weak Subjectives</h6>
                                </div>
                                <p className='the-answer-list-paragraph'>Adjectives or adverbs that add (subjective) force to the meaning of a phrase or proposition.
                                    <p>... did the <strong>fantastic</strong> reproductions... (subjective intensifier)</p>
                                    <p>... did the accurate reproductions...</p>
                                </p>
                            </span>
                        </span>
                    </span>
                    <span className='one-entry'>
                        <img style={{width:'50vw', margin:'auto'}}src={ sourceList } alt="Sources" />
                    </span>
                </div>
            </>
        )
    }
}