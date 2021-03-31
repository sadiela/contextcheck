import React, { Component } from 'react';
import Header from '../Homepage/Header/HomeButtonRow';

export default class AboutUs extends Component {
    render() {
        return(
            <>
                <Header />
                <h1>About Us</h1>
                <div className='full-deets-wrapper'>
                    <span className='one-entry'>
                        <h2 className='the-question'>
                            What is bias?
                        </h2>
                        <span className='the-answer-body'>
                            <p className='answer-paragraph'>
                                In general, it can be defined as prejudice in favor of or against one thing, person, or group compared with another, usually in a way considered to be unfair. When we read, watch, or listen to the news, we want to get an unbiased, or objective, view of whatever story is being presented. This is important because how we interpret the news affects how we vote, what we think of people, places, and businesses, and so many more things. When making decisions like these, we should be well informed.
                            </p>
                        </span>
                    </span>
                    <span className='one-entry'>
                        <h2 className='the-question'>
                            How does bias manifest in the news?
                        </h2>
                        <span className='the-answer-body'>
                            <p className='answer-paragraph'>
                                Bias can occur at many different levels in a news outlet or website. Here are some questions you can ask yourself to try to spot bias at the different levels.
                            </p>
                            <span className='answer-list-wrapper'>
                                <div className='title-wrapper'>
                                    <h4 className='answer-list-title'>Outlet Level</h4>
                                </div>
                                <ul className='answer-list'>
                                    <li className='answer-list-item'>What stories does the outlet cover? Do they align with a particular business or political agenda?</li>
                                    <li className='answer-list-item'>How articles are featured on websites? Which stories are placed at the very top? Which ones do you have to dig for?</li>
                                    <li className='answer-list-item'>When covering a story about a conflict, are both sides reported?</li>
                                </ul>
                            </span>
                            <span className='answer-list-wrapper'>
                                <div className='title-wrapper'>
                                    <h4 className='answer-list-title'>Article Level</h4>
                                </div>
                                <ul className='answer-list'>
                                    <li className='answer-list-item'>What is the title of the article? Is it misleading? </li>
                                    <li className='answer-list-item'>What images were chosen to accompany the story?</li>
                                    <li className='answer-list-item'>What perspectives are used in the article?</li>
                                    <li className='answer-list-item'>Are there more sources supporting one viewpoint than the other?</li>
                                </ul>
                            </span>
                            <span className='answer-list-wrapper'>
                                <div className='title-wrapper'>
                                    <h4 className='answer-list-title'>Sentence / Word Level</h4>
                                </div>
                                <ul className='answer-list'>
                                    <li className='answer-list-item'>What type of words does the author use?</li>
                                    <li className='answer-list-item'>Do the words have a positive/negative spin? eg. <p>Positive: "He is <strong>svelte</strong>."</p> <p>Negative: "He is <strong>skinny</strong>."</p></li>
                                    <li className='answer-list-item'>Are the words highly emotional? eg. <p>"Lee <strong>begged</strong> them for help."</p> <p>"Lee <strong>pestered</strong> them for help."</p> <p>"Lee <strong>asked</strong> them for help."</p></li>
                                </ul>
                            </span>
                        </span>
                    </span>
                    <span className='one-entry'>
                        <h2 className='the-question'>
                            What bias is relevant to ContextCheck?
                        </h2>
                        <span className='the-answer-body'>
                            <p className='answer-paragraph'>
                                Fighting bias in the news is a massive problem, so instead of boiling the ocean, we have chosen to tackle one small part. We focus on sentence and word level bias, and even more specifically, we focus on detecting inappropriate subjectivity. Inappropriate subjectivity occurs when language that should be neutral and fair is skewed by feeling, opinion, or taste (whether consciously or unconsciously). Although we would love to tackle additional levels of bias in the future (and bias/misinformation in social media), it was out of scope for this particular project.
                            </p>
                        </span>
                    </span>
                    <span className='one-entry'>
                        <h2 className='the-question'>
                            Why do we need ContextCheck?
                        </h2>
                        <span className='the-answer-body'>
                            <p className='answer-paragraph'>
                                The proportion of Americans who read and interact with news in an online format has been rapidly increasing in recent years[1]. Companies who curate online news content such as Facebook, Twitter, and Google use machine learning algorithms to maximize the amount of time that users spend on their platform. These algorithms often push false, misleading, and extreme information that reinforces people’s existing views and increases political polarization[2]. The goal of ContextCheck is to defuse the increasing societal tension inflamed by social media algorithms with another algorithm. 
                            </p>
                        </span>
                    </span>
                    <span className='one-entry'>
                        <h2 className='the-question'>
                            What is ContextCheck?
                        </h2>
                        <span className='the-answer-body'>
                            <p className='answer-paragraph'>
                                Context Check is a web application with the goal of promoting more informed readers of online news. It does that by providing context for a given article at three levels.
                            </p>
                            <span className='answer-list-wrapper'>
                                <div className='title-wrapper'>
                                    <h4 className='answer-list-title'>Metadata</h4>
                                </div>
                                <p className='the-answer-list-paragraph'>For a selected article, we provide important metadata including author, date of publication, and type of publication (e.g. news, opinion, satire). Providing this information to users will help avoid misunderstandings such as mistaking a satire source as a news source or citing something outdated.</p>
                            </span>
                            <span className='answer-list-wrapper'>
                                <div className='title-wrapper'>
                                    <h4 className='answer-list-title'>Related Articles</h4>
                                </div>
                                <p className='the-answer-list-paragraph'>Recommendation algorithms are rewarded for showing a user content similar to what they have already viewed and enjoyed, effectively creating “news bubbles” that can be hard to escape [3]. Seeing how other sources are reporting on the same issue can shed light on the shortcomings, emphasis, or omissions made in the article being tested.</p>
                            </span>
                            <span className='answer-list-wrapper'>
                                <div className='title-wrapper'>
                                    <h4 className='answer-list-title'>Bias Detection</h4>
                                </div>
                                <p className='the-answer-list-paragraph'>The final arm of ContextCheck is a machine learning model that detects biased words and phrases appearing in an article. This feature will alert users about specific linguistic features of articles that they should look out for, as they indicate subjectivity in the reporting.</p>
                            </span>
                        </span>
                    </span>
                    <span className='one-entry'>
                        <h2 className='the-question'>
                            Who are we?
                        </h2>
                        <span className='the-answer-body'>
                            <p className='answer-paragraph'>
                            ContextCheck was created by a senior design group from Boston University’s College of Engineering. Our team consists of the following four individuals.
                            </p>
                            <span className='answer-list-wrapper'>
                                <div className='title-wrapper'>
                                    <h4 className='answer-list-title'>Sadie Allen</h4>
                                </div>
                                <p className='the-answer-list-paragraph'>Sadie is majoring in computer engineering in mathematics and plans to pursue a PhD in computer engineering at BU after graduating in May. She enjoys running and experimenting with music production.</p>
                            </span>
                            <span className='answer-list-wrapper'>
                                <div className='title-wrapper'>
                                    <h4 className='answer-list-title'>Ye Chen</h4>
                                </div>
                                <p className='the-answer-list-paragraph'>Ye is majoring in computer engineering with a focus on software and looks to pursue education after graduating in May. He loves building custom computers and investing in stocks.</p>
                            </span>
                            <span className='answer-list-wrapper'>
                                <div className='title-wrapper'>
                                    <h4 className='answer-list-title'>Bayard Eton</h4>
                                </div>
                                <p className='the-answer-list-paragraph'>Bayard is majoring in computer engineering. He'll join the workforce as a software engineer after graduation, but he is planning to pursue an advanced degree in 2-3 years. He is a huge Boston sports fan, loves to meditate, and enjoys tossing a ball with the boys on a warm summer’s eve.</p>
                            </span>
                            <span className='answer-list-wrapper'>
                                <div className='title-wrapper'>
                                    <h4 className='answer-list-title'>Sean McDonald</h4>
                                </div>
                                <p className='the-answer-list-paragraph'>Sean is majoring in computer engineering with a concentration in Technology Innovation. He is going to work in software development in the greater Boston area after graduation. In his spare time he also likes to play electric bass and ride a unicycle (not at the same time).</p>
                            </span>
                        </span>
                    </span>
                    <span className='disclaimer-wrapper'>
                        <h5 className='the-question'>Disclaimers / Limitations</h5>
                        <span className='disclaimer'>
                            <h6 className='disclaimer-title'>ContextCheck is <strong>NOT</strong> a fact-checker.</h6>
                            <p className='disclaimer-paragraph'>We cannot determine the inherent truth of a given statement, nor is that our goal. We are simply detecting a certain style of language that should not be present in objective news reporting.</p>
                        </span>
                        <span className='disclaimer'>
                            <h6 className='disclaimer-title'>ContextCheck’s bias detection module is designed for use with <strong>news reporting</strong>.</h6>
                            <p className='disclaimer-paragraph'>When used with other types of writing (e.g. novels, creative writing, etc.), the results are not as relevant, since subjective language has an important place in many other genres. Holding writing to a standard of objectivity is most important for news reporting or scientific writing because of the goal of unbiased fact reporting.</p>
                        </span>
                        <span className='disclaimer'>
                            <h6 className='disclaimer-title'>ContextCheck would <strong>never</strong> be used to limit or block access to information</h6>
                            <p className='disclaimer-paragraph'>The goal is just to provide people with greater context for whatever news they choose to consume.</p>
                        </span>
                    </span>
                </div>
            </>
        )
    }
}