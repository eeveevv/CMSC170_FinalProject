const CONTENT = {
    title: "kAInan: Predicting Next Month Provincial Rice Prices in the Philippines with Ridge Regression, kNN, and MLP",
    members: 'Alyssa Kristelle Dela Cruz, Avigail Ann A. Monsalud, Natasha Lois S. Montas, Rex Robert L. Roxas<br>',

    links: `
        <a href="https://github.com/eeveevv/CMSC170_FinalProject.git" target="_blank">Github Code</a><br>
        <a href="https://canva.link/vhqm3nkaetpkmc8" target="_blank">Milestone 1 Slides</a><br>
        <a href="https://canva.link/hba0w8kt4syrapw" target="_blank">Milestone 2 Slides</a><br>
        <a href="https://canva.link/ubhmlb5n2a30ivx" target="_blank">Milestone 3 Slides</a>
    `,

    abstract: `
        <p>Food security is a common problem among Filipino households, mainly due to unaffordability. Among the possible foods that Filipinos can afford, rice is a staple, as it takes up much of their diet. Thus, predicting the next month’s price of rice can help households, local government units (LGUs), and retailers on their budgeting and purchases. This project assesses the ability of different machine learning models to predict the next month’s price of rice.</p>
        <p>The project uses three different models, k-Nearest Neighbor Regression, Ridge Regression, and Multilayer Perceptron to predict the next month’s price of rice. These models were compared to a baseline persistence model for its prediction. The models were tuned on 2025 data and tested once on January–August 2026. On the test set, the Multilayer Perceptron performed best (MAE ₱1.83/kg, R2 0.748) and was the only model to beat the baseline on every metric. It was followed by the baseline (MAE ₱2.00/kg), Ridge Regression (MAE ₱2.17/kg), and k-Nearest Neighbors (MAE ₱3.03/kg).</p>
    `,

    rationale: `
        <h3>Food Security and Food Access</h3>

        <p>Food Security, as defined by the World Food Summit in 1996, “exists when all people, at all times, have physical and economic access to sufficient, safe and nutritious food to meet their dietary needs and food preferences for an active and healthy life” [1]. This definition also defines four dimensions of food security: availability, access, utilization, stability. In particular, food access, by definition, is the individual’s access to adequate resources for acquiring appropriate foods for a nutritious diet, given their legal, political, social, and economic background [2].</p>

        <h3>Food Access in Philippines</h3>

        <p>In the Philippines, it was found that many Filipino households are unable to consume nutrient-adequate diets, mainly due to unaffordability [3]. Around one out of three Filipino households cannot afford a nutrient-adequate diet [4].</p>

        <h3>Rice as a Staple Food</h3>

        <p>Among the common household foods in the Philippines, rice remains a staple in Filipino households, with rice being 35.7% of the average Filipino household’s daily food intake, which contributes to 58.0% of the total energy intake for Filipinos [5].</p>

        <h3>Why Predict Rice Prices</h3>

        <p>A reliable estimate of next month’s rice price in each province can help households, local government units, and retailers plan their budgets and purchases ahead of time. This project tests whether machine learning models built from scratch can give a better estimate than simply assuming the price will stay the same.</p>
    `,

    problem: `
        <p class="callout">
            <p>Can we predict next month’s retail rice price (₱/kg) in a province from its recent prices, the national price trend, and the time of year?</p>
        </p>
    `,

    dataset: `
        <p>The dataset used in this project is “Philippines - Food Prices (World Food Programme)”, published by the World Food Programme on the Humanitarian Data Exchange [6]. It contains monthly market prices of food items across the Philippines.</p>
    `,

    methods: `
        <p>The models used in this project are K-Nearest Neighbor Regression, Ridge Regression, and Multilayer Perceptron.</p>

        <h3>K-Nearest Neighbors</h3>

        <p>K-Nearest Neighbor estimates the target variable using an average of the K-Nearest Neighbors.</p>
        <p>K-Nearest Neighbors fits because the dataset includes records of prices for different provinces of the different months from the previous years. For a new month, it can find past months that looked similar (eg., a similar national price, recent price change, and price gap), and predicts the retail price of the milled rice will move the way it did after those months.</p>
        <p>A previous study predicted which news topics would trend on Twitter before Twitter declared them trending, 79% of the time, with a 95% true positive rate and 4% false positive rate [7]. Since kernel classifiers, like kNN, predict from similarly labeled history, this supports using neighbor-based methods on our dataset [8].</p>

        <h3>Ridge Regression</h3>

        <p>Ridge Regression estimates the target variable using a weighted sum of inputs. A regularization term (also called as penalty) keeps the weights from getting too large, which reduces overfitting [9]. </p>
        <p>Inputs have several features that could be highly correlated, such as past rice prices, last month’s change in the national average rice price, and the province’s price gap versus the national mean. This makes ridge regression suitable, as it keeps the weight small which is best for multicollinearity [10].</p>

        <h3>Multilayer Perceptron</h3>

        <p>Multilayer Perceptrons is a fundamental model, which uses a series of hidden weights to compute the output variable from the input variables [11].</p>
        <p>The Multilayer Perceptron is a powerful improvement to the Perceptron, as it can capture non-linear decision boundaries and can be used in regression analysis [11]. The features in this study have varying scales and relationships, which may make it too complex for a regular Perceptron. With Multilayer Perceptrons, complex relationships within data that cannot be described by a linear decision boundary can be learned and used to generate more accurate predictions based on our data.</p>
    `,

    setup: `
        <p>The models were created and trained in Python, using NumPy for the linear algebra operations, math for operations not involving linear algebra, Matplotlib for the graphs, and csv for importing of data. The experiment had four steps:</p>

        <ol>
            <li><strong>Preprocessing</strong>
                <ul>
                    <li>The data was extracted from the dataset from the World Food Programme retrieved on October 27, 2026.</li>

                    <li>The data was cleaned, and the data points for the regular milled rice were taken.</li>

                    <li>The data was filtered through these criteria:
                        <ol type="i">
                            <li>The data was from 2020–2026.</li>
                            <li>The data was for retail prices.</li>
                            <li>The data was from places with a complete record of the prices of rice.</li>
                        </ol>
                    </li>

                    <li>Several features were created:
                        <ol type="i">
                            <li>National average rice price in each of the previous 3 months</li>
                            <li>Last month’s change in the national average rice price</li>
                            <li>Province’s price gap versus the national mean</li>
                            <li>Month</li>
                        </ol>
                    </li>

                    <li>The data was split by date into a training set (May 2020–December 2024), a validation set (2025), and a test set (January–August, 2026).</li>
                </ul>
            </li>

            <li><strong>Tuning</strong>
                <ul>
                    <li>Each hyperparameter setting was trained on the training set and scored on the validation set. The setting with the best overall performance among the chosen evaluation metrics was kept.</li>

                    <li>The hyperparameters tuned for the K-Nearest Neighbors are the number of nearest data points <em>k</em> and the Minkowski distance order <em>p</em>.</li>

                    <li>For Ridge Regression, it is the penalty constant λ.</li>

                    <li>For Multilayer Perceptron, it is the learning rate per epoch, number of layers and their sizes, and the number of epochs.</li>
                </ul>
            </li>

            <li><strong>Validation</strong>
                <ul>
                    <li>With the chosen settings, each model was trained on 2020–2024 and evaluated on 2025.</li>
                </ul>
            </li>

            <li><strong>Final test</strong>
                <ul>
                    <li>After tuning, each model was trained on the training set only (2020–2024) with its chosen settings, and evaluated once on the test set (January–August 2026).</li>
                </ul>
            </li>
        </ol>
    `,

    baseline: `
        <p>Each model was compared against a persistence baseline, which predicts that next month’s price will be the same as the market’s latest observed price. The baseline needs no training or tuning. It was scored on the same validation and test sets with the same metrics.</p>
    `,

    eval_metrics: `
        <p>The models were evaluated with four metrics, shown in Table 1.</p>

        <p><b>Table 1.</b> Evaluation metrics</p>

        <table>
        <tr><th>Metric</th><th>Unit</th><th>What it tells us</th></tr>
        <tr><td>Mean Absolute Error (MAE)</td><td>₱/kg</td><td>On average, how far off is the ₱/kg prediction</td></tr>
        <tr><td>Root Mean Squared Error (RMSE)</td><td>₱/kg</td><td>Similar to MAE, but big misses have a heavier effect</td></tr>
        <tr><td>Mean Absolute Percentage Error (MAPE)</td><td>%</td><td>The average error as a percentage of the actual price</td></tr>
        <tr><td>R-squared or Coefficient of Determination (R²)</td><td>none</td><td>The share of the variation in rice prices that the model explains. 1 is perfect, and 0 is no better than always predicting the average price</td></tr>
        </table>

        <p>We also recorded the time each model took to train and produce its predictions.</p>
        `,

    results: `
        <h3>Validation Set</h3>

        <p>Table 2 presents the evaluation metrics for each model on the validation set (2025).</p>

        <p><b>Table 2.</b> Validation set results (2025)</p>

        <div class="table-wrap">
            <table>
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>MAE (₱/kg)</th>
                        <th>RMSE (₱/kg)</th>
                        <th>MAPE</th>
                        <th>R²</th>
                        <th>Time (s)</th>
                    </tr>
                </thead>
                <tbody>
                    <tr><td>Baseline</td><td>1.1871</td><td>1.8138</td><td>2.8096</td><td>0.8388</td><td>0.0000</td></tr>
                    <tr><td>Ridge Regression</td><td>1.6819</td><td>2.1646</td><td>4.0171</td><td>0.7704</td><td>0.0010</td></tr>
                    <tr><td>k-Nearest Neighbor</td><td>2.1280</td><td>2.7559</td><td>5.0160</td><td>0.6278</td><td>2.4212</td></tr>
                    <tr><td>Multilayer Perceptron</td><td>1.6778</td><td>2.2185</td><td>4.0269</td><td>0.7588</td><td>0.0045</td></tr>
                </tbody>
            </table>
        </div>

        <p>The baseline model performs the best on the validation set, with the lowest MAE (1.1871), RMSE (1.8138), and MAPE (2.8096%) and the highest R² (0.8388). Ridge Regression performs second best, with a better RMSE (2.1646), MAPE (4.0171%), and R² (0.7704) than the Multilayer Perceptron. It is only slightly worse on MAE (1.6819 vs. 1.6778). The Multilayer Perceptron follows closely, with an RMSE of 2.2185, a MAPE of 4.0269%, and an R² of 0.7588. K-Nearest Neighbors performs the worst, with an MAE of 2.1280, RMSE of 2.7559, MAPE of 5.0160%, and R² of 0.6278.</p>

        <h3>Test Set</h3>

        <p>Table 3 presents the evaluation metrics for each model on the test set (January–August 2026).</p>

        <p><b>Table 3.</b> Test set results (January–August 2026)</p>

        <div class="table-wrap">
            <table>
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>MAE (₱/kg)</th>
                        <th>RMSE (₱/kg)</th>
                        <th>MAPE</th>
                        <th>R²</th>
                        <th>Time (s)</th>
                    </tr>
                </thead>
                <tbody>
                    <tr><td>Baseline</td><td>2.0003</td><td>2.9083</td><td>4.4154</td><td>0.6446</td><td>0.0000</td></tr>
                    <tr><td>Ridge Regression</td><td>2.1664</td><td>2.7760</td><td>4.8400</td><td>0.6762</td><td>0.0010</td></tr>
                    <tr><td>k-Nearest Neighbor</td><td>3.0288</td><td>4.1244</td><td>6.8626</td><td>0.2853</td><td>1.6631</td></tr>
                    <tr><td>Multilayer Perceptron</td><td>1.8333</td><td>2.4479</td><td>4.0325</td><td>0.7482</td><td>0.0091</td></tr>
                </tbody>
            </table>
        </div>

        <p>The Multilayer Perceptron performs the best on the test set, with the lowest MAE (1.8333), RMSE (2.4479), and MAPE (4.0325%) and the highest R² (0.7482). The baseline is second on MAE (2.0003) and MAPE (4.4154%), while Ridge Regression is second on RMSE (2.7760) and R² (0.6762), which means it makes fewer large errors than the baseline. K-Nearest Neighbors performs the worst again, with an MAE of 3.0288, RMSE of 4.1244, MAPE of 6.8626%, and R² of 0.2853. All models except the Multilayer Perceptron have higher errors on the test set than on the validation set.</p>
    `,

    discussion: `
        <p>The baseline model performed well on the validation data, but performed worse in the test dataset. This can be attributed to the data in the validation dataset having less fluctuation in prices across months, and thus using the previous month for predicting the current month’s prices will result in a better performance. However, it is possible that there is a large fluctuation in the price of rice in the test dataset, and thus using the previous month for predicting the current month’s prices will not be as accurate.</p>

        <p>The k-Nearest Neighbors performed badly on the dataset both in the validation and test sets. This may be due to the features being structured in a way that similar data points in terms of features do not have a similar price. In addition, since kNN predicts an average of past prices, it cannot predict prices higher than those in the training data.</p>

        <p>The Ridge Regression performed well on the validation set, but performed slightly worse on the test set. This may be because the relationship between the features and the output variables is easily defined linearly, especially in the validation phase. However, some relationships between the features and output variables were not as easily defined linearly as seen in its performance on the test data, possibly due to an external force that is not accounted for in the model, such as the sudden rise in the prices in early 2026.</p>

        <p>The Multilayer Perceptron performed well on the dataset in the validation and test metrics. This can be explained by the possible nonlinear relationship between the features and the predicted variables. Its error rose only slightly from validation to test (MAE 1.68 to 1.83), while the other models’ errors rose more, which suggests it adapted better to the change in the price trend.</p>
    `,

    conclusion: `
        <p>We built the K-Nearest Neighbors Regression, Ridge Regression, and a Multilayer Perceptron from scratch and compared them against a persistence baseline. The Multilayer Perceptron performed best on the test set, and it was the only model to beat the baseline on every metric. Ridge Regression was the most consistent and fastest to train, while K-Nearest Neighbors performed the worst because it cannot predict outside the range of past prices. For future work, the models could use each market’s own past prices, outside factors such as weather and imports, and validation over several years.</p>
    `,

    references: `
        <ol>
            <li>World Food Summit, “Rome Declaration and Plan of Action,” <em>www.fao.org</em>, 1996. <a href="https://www.fao.org/4/w3613e/w3613e00.htm" target="_blank">https://www.fao.org/4/w3613e/w3613e00.htm</a> (accessed Oct. 27, 2026).</li>

            <li>Food and Agriculture Organisation, “Food Security Policy Brief,” June 2006. Accessed: Oct. 27, 2026. [Online]. Available: <a href="https://www.fao.org/fileadmin/templates/faoitaly/documents/pdf/pdf_Food_Security_Cocept_Note.pdf" target="_blank">https://www.fao.org/fileadmin/templates/faoitaly/documents/pdf/pdf_Food_Security_Cocept_Note.pdf</a></li>

            <li>I. M. Galang, “Is Food Supply Accessible, Affordable, and Stable? The State of Food Security in the Philippines,” June 2022, doi: 10.62986/dp2022.21.</li>

            <li>“FF2026-02: PHILIPPINE RICE SUPPLY 2018 to 2024 | CPBRD,” <em>Congress.gov.ph</em>, 2026. <a href="https://cpbrd.congress.gov.ph/ff2026-02-philippine-rice-supply-2018-to-2024/" target="_blank">https://cpbrd.congress.gov.ph/ff2026-02-philippine-rice-supply-2018-to-2024/</a> (accessed Oct. 27, 2026).</li>

            <li>K. A. Egalin, “Dost.Gov.Ph - DOST-FNRI Unveils 2023 Filipinos State of Health and Nutrition,” Dost.gov.ph, July 03, 2025. <a href="https://www.dost.gov.ph/knowledge-resources/news/86-2025-news/4067-dost-fnri-unveils-2023-filipinos-state-of-health-and-nutrition.html" target="_blank">https://www.dost.gov.ph/knowledge-resources/news/86-2025-news/4067-dost-fnri-unveils-2023-filipinos-state-of-health-and-nutrition.html</a> (accessed Oct. 27, 2026).</li>

            <li>World Food Programme, “Philippines – Food Prices,” Humanitarian Data Exchange. Accessed: Oct. 27, 2026. [Online]. Available: <a href="https://data.humdata.org/dataset/wfp-food-prices-for-philippines" target="_blank">https://data.humdata.org/dataset/wfp-food-prices-for-philippines</a></li>

            <li>S. Nikolov and D. Shah, “A nonparametric method for early detection of trending topics,” presented at the Interdisciplinary Workshop on Information and Decision in Social Networks, 2012.</li>

            <li>G. H. Chen and D. Shah, “Explaining the Success of Nearest Neighbor Methods in Prediction,” <em>Foundations and Trends in Machine Learning</em>, vol. 10, no. 5–6, pp. 337–588, May 2018, doi: 10.1561/2200000064.</li>

            <li>J. Murel and E. Kavlakoglu, “What is ridge regression?,” IBM, Sep. 21, 2023. [Online]. Available: <a href="https://www.ibm.com/think/topics/ridge-regression" target="_blank">https://www.ibm.com/think/topics/ridge-regression</a></li>

            <li>T. Hastie, R. Tibshirani, and J. Friedman, <em>The Elements of Statistical Learning: Data Mining, Inference, and Prediction</em>, 2nd ed. New York, NY, USA: Springer, 2009, sec. 3.4.1.</li>

            <li>D. Saini, “Deep Learning Specialization: Understanding Multi-Layer Perceptron (MLP) Inside Out,” Medium, Aug. 31, 2025. <a href="https://medium.com/data-science-collective/deep-learning-specialization-understanding-multi-layer-perceptron-mlp-inside-out-f3ff465a521f" target="_blank">https://medium.com/data-science-collective/deep-learning-specialization-understanding-multi-layer-perceptron-mlp-inside-out-f3ff465a521f</a> (accessed Oct. 27, 2026).</li>

        </ol>
    `,
    disclosure: `
        <p>All models and evaluation metrics were implemented from scratch using only NumPy, Matplotlib, and the Python standard library, without any machine learning libraries. The online sources we learned from are cited in the References and in the code comments. We also used Claude (Anthropic) to explain concepts and review our code and reports. All AI-assisted code and text were checked, run, and edited by the group, and all reported results come from running our own code.</p>
    `,
};