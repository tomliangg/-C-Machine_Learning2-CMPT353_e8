﻿# Machine_Learning2-CMPT353_e8
<h3>This repo is created for documentation purpose. The repo contains my personal work toward the SFU CMPT353 (Computational Data Science) course. You may use my solution as a reference. The .zip archive contains the original exercise files. For practice purpose, you can download the .zip archive and start working from there.</h3>

<p><a href="https://coursys.sfu.ca/2018su-cmpt-353-d1/pages/AcademicHonesty">Academic Honesty</a>: it's important, as always.</p>

<br/>
<p>There are 3 tasks in this exercise. </p>
<p>First task: it's quite similar to exercise-seven's task 3&4 but you are expected to build more models (Support Vector Machine, K-Nearest Neighbours, plus Naive Bayes). You also need to compare the scores for different classifiers.</p>
<p>Second task: given 60 weather features with labels, you need to use classification (supervised learning) and train a model to predict for the unlabelled weather data.</p>
<p>Third task: instead of using classification, you will play around with clustering (unsupervised learning algorithms). The clustering technique can help find observations with similar weathers. With the scatter plot and clustering group, you will understand why the model you built in 2nd task could make prediction mistakes. </p>
<br/>

<p>Below is the exercise description </p>
<hr>

<h2 id="h-colour-words-again">Colour Words, Again</h2>
<p>Last week, we used a Na<span>&iuml;</span>ve Bayesian classifier to do the RGB values to colour words task. Since then, we have explored some more techniques: <em>k</em>-nearest neighbours and SVM classifiers. Let's compare techniques<span>&hellip;</span></p>
<p>Create a program <code>colour_predict.py</code> that takes the input CSV file on the command line, as last week. A <strong>new</strong> hint has been included this week to help get your input/output into the right shape.</p>
<p>Include your <code>GaussianNB</code>-based classifiers from last week, so we can compare: You should have a model that gives the original RGB colours to the classifier, and one that converts to LAB colours and then trains the classifier.</p>
<p>Do the same with a <em>k</em>-nearest neighbours classifier (<code>KNeighborsClassifier</code>). For both the RGB and LAB version, tune the <em>k</em> parameter to get the best results.</p>
<p>Finally, do the same with an SVM classifier (<code>SVC</code>), adjusting the <em>C</em> parameter for best results.</p>
<p>When finished, your <code>colour_predict.py</code> should <strong>print the scores on your test data in the format included in the hint</strong>. Please do <strong>not</strong> have a <code>plt.show()</code> in your code when you submit: it makes marking a pain.</p>
<h2 id="h-case-of-the-unlabelled-weather">Case of the Unlabelled Weather</h2>
<p>We have met the <a href="https://www.ncdc.noaa.gov/data-access/land-based-station-data/land-based-datasets/global-historical-climatology-network-ghcn">GHCN</a> data before<span>&hellip;</span> but oh no! When recording the 2016 data, somebody forgot to record <em>which city</em> the observations came from. Whatever will we do? *</p>
<p>For this question, I took data from 26 North American airports and extracted several features: daily minimum and maximum temperature (in 0.1<span>&deg;</span>C), amount of precipitation (in 0.1mm), snowfall (in mm), and snow depth (in mm). For each feature, I calculated the monthly average for each month of the year, for a total of 60 features for each city and year.</p>
<p>In the provided <code>monthly-data-labelled.csv</code>, you will find all of these features, as well as the name of the city and year of the observations. The file <code>monthly-data-unlabelled.csv</code> is the same, but with the city name redacted: that's what we're hoping to reconstruct.</p>
<p>Create a program <code>weather_city.py</code> that reads the labelled data and trains and tests a machine learning model for the best possible results. It should then predict the cities where the unlabelled 2016 weather came from.</p>
<p>The command line should take filenames for the labelled, unlabelled, and output files:</p>
<pre class="highlight lang-bash">python3 weather_city.py monthly-data-labelled.csv monthly-data-unlabelled.csv labels.csv</pre>
<p><strong>The output format</strong> (into the file given as the third command line argument) should be one city name per line, in the format this line produces:</p>
<pre class="highlight lang-python">pd.Series(predictions).to_csv(sys.argv[3], index=False)</pre>
<p>Your program should <strong><code>print</code> one line</strong>: the <span>&ldquo;</span>score<span>&rdquo;</span> of the model you're using on a testing subset of the labelled data.</p>
<p>The features have very different magnitudes: the features with units 0.1<span>&deg;</span>C maximum temperature and millimetres of snow are nowhere close to the same scale. You'll probably want to normalize into a predictable range. <a href="http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html">Hint</a>.</p>
<p>* Nobody from the GHCN forgot to record what city data originated. They are all fine, upstanding data collectors.</p>
<h3 id="h-exploring-the-weather">Exploring the Weather</h3>
<p>Why did that work? How was a machine learning model able to take the weather observations and (usually) come up with the correct city? We can explore the data a little more to get a sense of its structure.</p>
<p>See the attached <code>weather_clusters_hint.py</code> which (when completed) can be run like:</p>
<pre class="highlight lang-bash">python3 weather_clusters.py monthly-data-labelled.csv</pre>
<p>Start with the same <code>X</code> and <code>y</code> values you used in the previous part: all of the observations, and the correct cities.</p>
<p>We will first use principal component analysis to get two-dimensional data that we can reasonably plot. Fill in the provided <code>get_pca</code> function so it returns the <code>X</code> data transformed to its two most <span>&ldquo;</span>important<span>&rdquo;</span> features. Hint: <a href="http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html">MinMaxScaler</a> seems to work better here, and of course you'll need the <code>PCA</code> model.</p>
<p>We can also use a clustering technique to find observations with similar weather. Fill in the provided <code>get_clusters</code> to find 10 clusters of similar weather observations using <a href="http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html">KMeans</a>  clustering.</p>
<p>With that, you should get a scatter plot of the clusters: note that none of the input to the plot used the <code>y</code> values. It was created just by examining the observed <code>X</code>  values.</p>
<p>The provided code also creates and prints a table of how many observations from each city were  put into each category (using the <code>y</code> values now). You should be able to see here which cities have similar weather patterns.</p>
<h2 id="h-questions">Questions</h2>
<p>Answer these questions in a file <code>answers.txt</code>.</p>
<ol><li>Which model did the best for the colour-prediction task? Can you give a theory about why?
</li><li>Have a look at the cities in your test data where the weather model makes the wrong prediction. Do you feel like the model is making reasonable mistakes? Can you think of any weather features that we could potentially add to make better predictions? 
</li></ol>
<p>Here's a hint for that, but please <strong>comment-out</strong> the <code>print</code> before submitting, so we don't have to wade through the output. I'm not as concerned about your answer here as you looking at the predictions your model makes and evaluating it with a human-brain-based critique, not just an accuracy score.</p>
<pre class="highlight lang-python">df = pd.DataFrame({'truth': y_test, 'prediction': model.predict(X_test)})
print(df[df['truth'] != df['prediction']])</pre>
