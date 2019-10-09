## Virtual Materials Intelligence for Design and Discovery of Advanced Electrocatalysts 

Ref. [1]: Eslamibidgoli, Mohammad Javad, Mehrdad Mokhtari, and Michael H. Eikerling. "Recurrent Neural Network-based Model for Accelerated Trajectory Analysis in AIMD Simulations." arXiv preprint arXiv:1909.10124 (2019).

Ref. [2]: Virtual Materials Intelligence for Design and Discovery of Advanced Electrocatalysts (In proof: DOI: 10.1002/cphc.201900570R1)


# Abstract

Similar to the advancements gained from big data in genomics, security, internet of things, and e-commerce, the materials workflow could be made more efficient and prolific through advances in streamlining data sources, autonomous materials synthesis, rapid characterization, big data analytics, and self-learning algorithms. In electrochemical materials science, data sets are large, unstructured/heterogeneous, and difficult to process and analyze from a single data channel or platform. Computer-aided materials design together with advances in data mining, machine learning, and predictive analytics are touted to provide inexpensive and accelerated pathways towards tailor-made functionally optimized energy materials. Fundamental research in the field of electrochemical energy materials focuses primarily on complex interfacial phenomena and kinetic electrocatalytic processes. This perspective article critically ranks existing AI-driven modeling and computational approaches that are currently applied to those objects. An application-driven materials intelligence platform is also introduced, and its functionalities are scrutinized considering the development of electrocatalyst materials for CO2 conversion as a use case.

# 1. Introduction
				
Machine learning is a highly active branch in computer science that consists of a set of algorithms that find patterns in a given data. Data is expressed in the form of vectors associated to a (high dimensional) feature space, and the common task of a machine learning algorithm (whether it is a generative or a discriminative model) is to "learn" from data and make predictions on unseen data. The learning process typically involves an optimization routine, such as stochastic gradient descent, to find a set of hidden parameters associated to features that minimize an objective function (e.g., minimizing the negative of log-likelihood function). Given the parameters, the predictive model can be appiled to unseen data. Although the primary methods were developed in 1950s-1980s, there has only been many recent breakthrough in aritificial intelligence with interesting appications such as in computer vision, speech and language technologies, self-driving cars, recommender systems, financial predictions, robotics, etc. This is mainly due to 1. more recent complicated developed arhcitecture and algorithms for neural nets, 2. availability of much bigger data for model training, 3. much more powerful parallel computer processing, 4. enhanced frameworks for implementation, and 5. of course, larger industrial investments in the field. 
		
Extensive research is recently focused on employing such methods in theoretical chemistry for developing inter-atomic potentials to "bypassing the Kohn-Sham equations" (Brockherde_2017), thus, speeding up density functional theory (DFT) calculations, performing ab initio molecular dynamics (AIMD) simulation at a fraction of the cost of AIMD calculations (Chmiela_2017), as well in molecular design (Duvenaud_2015, Sanchez-Lengeling_2018), drug design (Gómez-Bombarelli_2018, Aspuru-Guzik_2018) or materials discovery (Jain_2013),  also with particular applications in energy storage and conversion devices (Jha_2017, Goldsmith_2018). In electrocatalysis, the goal is to find cheaper, more selective, more active and more durable catalyst materials for a desired electrochemical reaction; employing machine learning and quantum chemistry can significantly reduce the time for predicting/prioritizing such mateirlas. In this perspective, we first discuss the challenges in first-principles electrochemistry, followed by a brief overview on some of the most recent quantum mechanical-based machine learning models. In the final section, we introduce Virtual Materials Intelligence (VMI) Labs as an open-source platform for generating database for clean energy applications.          

Quantum mechanical calculations based on DFT are still the state-of-the-art methodology for understanding reaction mechanisms and predicting activity and selectivity of catalyst materials for applications in energy storage and conversion devices (Eslamibidgoli_2016). However, system size and complexity of the electrode-electrolyte interface and its immense parameter space limit such methods in terms of proper simulation of the interface, as well rapid materials screening (Eslamibidgoli_2018). System size exponentially scales the computational cost of DFT-based approaches which in turn limit the generalization and scalability of DFT methods. Moreover, complexity of the system involved in the electrode region (the structure, shape and composition of material), in the interfacial region (the water structure, adsorbed species, surface charging behavior), and in the electrolyte region (pH, ion distribution, and reactant distribution), play crucial roles in determinig the structure-property relations of the electrocatalyst material. Additionaly, reasonable sampling at the electrolyte side, i.e., statistical averaging over all possible configurations, is not feasible from first-principles methods alone. Therefore, DFT should be considered as a first (and essential step) in a hierarchy of methods to reasonably address these challenges. In this context, machine learning have been shown to be promising both in terms of predicting the potential energy surface of chemical structures, as well as fast discovery and design of target structures. In the following, we explain the typical approach for employing machine learning for materials discovery (Meredig_2014, Ward_2016, De Luna_2017).

A DFT-based machine learning approach requires a highly qualified dataset for training the model; thus, given a raw data for specific system, feature engineering is needed to be employed by the reseach scientist who has the domain knowledge. Therefore, the first step is to generate a combinatorial dataset using DFT, ab initio molecular dynamics (AIMD), or DFT-based Monte-Carlo simulations in order to compute mixture of structures (various surface states, nanoparticles, bulk structures, slabs, with water layers, etc.); an order of tens of thousands of feature vectors are needed, as the larger the dataset the better to train the ML model and to capture the statistics of the system. Data should account for variables for the electronic structure of the solid electrode, solvent properties and ion distributions in the electrolyte as well as specific properties of a boundary region in-between; such informative features should be provided in standardized form (mean as 0, standard deviation as 1). Appropriate encoding may be required to transform categorical features to numerial ones. To generate this large dataset efficient data harvesting method is needed along with human and computational resources.

Second step is to select the machine learning model and to train it for searching, classifying, or clustering the chemical space in terms of structure-property relationship. This is an efficient screening/filtering of structure-property relationships to prioritize the materials of interest (maybe only a hundred out of 100,000 are predicted as good) (Meredig_2014). Model selection (i.e. in classification, regression, clustering or dimensionality reduction methods) primarily depends on the size of data, (non)linearity of the data, whether it is labeled or not, memory and time efficiency (computational complexity) of the algorithm in training and prediction phases, its scalability as well its prediction accuracy. Appropriate techniques like random search or cross validation should be employed on training set and validation set to tune the parameters and hyper parameters of the model. Likewise, feature engineering is required to avoid high bias (underfitting problem) and regularization or dimensionality reduction techniques may be used to avoid high variance in the model (overfitting problem). 

Next step is to use DFT again to calculate the properties of the predicted structures and evaluate the performance of the machine learning predicted and DFT calculated structures (Pyzer-Knapp_2015, Hachmann_2011). Finally, proposed material should be experimentally synthesized and be tested (essential step); Consequently, out of a hundred of predicted materials maybe only a few of them are useful.


## 5.	References

1. Aspuru-Guzik, Alan, Roland Lindh, and Markus Reiher. ACS central science 4.2 (2018): 144-152.
2. Duvenaud, David K., et al. Advances in neural information processing systems. 2015.
3. Jain, Anubhav, et al. "Commentary: The Materials Project: A materials genome approach to accelerating materials innovation." Apl Materials 1.1 (2013): 011002.
4. Chmiela, Stefan, et al. "Machine learning of accurate energy-conserving molecular force fields." Science advances 3.5 (2017): e1603015.
5. Jha, Sunil Kr, et al. "Renewable energy: Present research and future scope of Artificial Intelligence." Renewable and Sustainable Energy Reviews 77 (2017): 297-317. 
6. Gómez-Bombarelli, Rafael, et al. ACS central science 4.2 (2018): 268-276., 
7. Sanchez-Lengeling, Benjamin, and Alán Aspuru-Guzik. Science 361.6400 (2018): 360-365.
8. Eslamibidgoli, Mohammad J., et al. "How theory and simulation can drive fuel cell electrocatalysis." Nano Energy 29 (2016): 334-361.
9. Eslamibidgoli, Mohammad J., and Michael H. Eikerling. "Approaching the self-consistency challenge of electrocatalysis with theory and computation." Current Opinion in Electrochemistry 9 (2018): 189-197.
10. Artrith, Nongnuch, and Alexie M. Kolpak. "Understanding the composition and activity of electrocatalytic nanoalloys in aqueous solvents: A combination of DFT and accurate neural network potentials." Nano letters 14.5 (2014): 2670-2676.
11. Groß, Axel, et al. Journal of The Electrochemical Society 161.8 (2014): E3015-E3020.
12. Meredig, Bryce, et al. "Combinatorial screening for new materials in unconstrained composition space with machine learning." Physical Review B 89.9 (2014): 094104., 
13. Ward, Logan, et al. "A general-purpose machine learning framework for predicting properties of inorganic materials." npj Computational Materials 2 (2016): 16028.
14. Pyzer-Knapp, Edward O., et al. Annual Review of Materials Research 45 (2015): 195-216., 
15. Hachmann, Johannes, et al. The Journal of Physical Chemistry Letters 2.17 (2011): 2241-2251.
16. Brockherde, Felix, et al. "Bypassing the Kohn-Sham equations with machine learning." Nature communications 8.1 (2017): 872.
17. Ulissi, Zachary W., et al. "To address surface reaction network complexity using scaling relations machine learning and DFT calculations." Nature communications 8 (2017): 14621.
18. Goldsmith, Bryan R., et al. "Machine learning for heterogeneous catalyst design and discovery." AIChE Journal 64.7 (2018): 2311-2323.
19. Yao, Kun, et al. "The TensorMol-0.1 model chemistry: a neural network augmented with long-range physics." Chemical science 9.8 (2018): 2261-2269.
20. Artrith, Nongnuch, and Alexie M. Kolpak. Nano letters 14.5 (2014): 2670-2676. 
21. Hy, Truong Son, et al. "Predicting molecular properties with covariant compositional networks." The Journal of chemical physics 148.24 (2018): 241745.
22. Behler, Jörg. Physical Chemistry Chemical Physics 13.40 (2011): 17930-17955.
23. Gómez-Bombarelli, Rafael, et al. "Automatic chemical design using a data-driven continuous representation of molecules." ACS central science 4.2 (2018): 268-276.
24. De Luna, Phil, et al. "Use machine learning to find energy materials." (2017): 23.
25. Brockherde, Felix, et al. "Bypassing the Kohn-Sham equations with machine learning." Nature communications 8.1 (2017): 872.
