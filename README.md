# Perspective: Machine learning for accelerated materials discovery in electrocatalysis

## Outline Summary

1.	Employing data-driven methods for accelerated materials discovery 
2.	A review section on recent data-driven methods for electro-catalyst design  
3.	Formulating an approach to speed up the materials discovery with machine learning
4.	Necessity for generating specialized data for specific reaction/catalyst   	
5.	Introducing Virtual Materials Intelligence (VMI) Labs as an open-source platform for collaborative research in materials discovery for clean energy applications 

## Abstract

## 1. Introduction
				
Machine learning is a branch in computer science that consists of a set of algorithms that find patterns in a given data. Data is expressed in the form of vectors belonging to a (high dimensional) feature space, and the common task of a machine learning algorithm (either it is a generative or a discriminative model) is to "learn" from data and make predictions on unseen data. Although the primary methods were developed in 1950s-1980s, there has only been many recent breakthrough in aritificial intelligence with interesting appications such as in computer vision, speech and language technologies, self-driving cars, recommender systems, financial predictions, robotics, etc. This is mainly due to 1. more recent complicated developed arhcitecture and algorithms for neural nets, 2. availability of much bigger data for model training, 3. much more powerful parallel computer processing, 4. enhanced frameworks for coding, and 5. of course, larger industrial investments in the field. 
		
Extensive research is recently focused on employing such methods in theoretical chemistry for developing inter-atomic potentials to "bypassing the Kohn-Sham equations" (Brockherde_2017) and speeding up density functional theory (DFT) calculations (cite alan aspuru NN paper 2015), developing of force-field for molecular dynamics simulations (cite a few), as well in molecular design, drug design or materials discovery with applications in energy storage and conversion devices (Aspuru-Guzik_2018, Duvenaud_2015, Jain_2013, Agrawal_2016, Jha_2017, Gómez-Bombarelli_2018, Sanchez-Lengeling_2018). As the main goal in electrocatalysis is to find cheaper, more selective, more active and more durable electrocatalyst materials for a desired electrochemical reaction, employing machine learning and quantum chemistry can significantly reduce the time for prioritizing such mateirlas. In the following, we first discuss the achievements and challenges in first-principles electrochemistry. This is followed by a brief overview on recent quantum mechanical-based machine learning models developed for screening and prioritizing molecules/materials. In the final section, we introduce Virtual Materials Intelligence (VMI) Labs as an open-source platform for generating database for clean energy applications.          


Quantum mechanical calculations based on DFT have been extensively employed in predicting activity and selectivity of catalyst materials for applications in energy storage and conversion devices (Eslamibidgoli_2016). However, due to complexity of the electrode-electrolyte interface and its immense parameter space, DFT methods are inherently limited for fast predictions (Eslamibidgoli_2018); in the electrode region, the structure, shape and composition of material, in the interfacial region, the water structure, adsorbed species, surface charging behavior, and in the electrolyte region, pH, ion distribution, and reactant distribution play crucial roles in determinig the structure-property relation of the electrocatalyst material. Main challenges include (1) construct continuous path for potential between metal bulk and electrolyte bulk (across all components, layers and interface); while DFT functionals (qm and classical) describe bulk properties well, coupling of regions at interfaces not well represented in interface functionals (functionals for adsorbates, functionals for surface solvent layers…). Therefore, is it possible at all to do this completely self-consistently from first principles or will there have to be some layer of empirical input? Example: oxide layer formation at Pt… Illustration: inability to reproduce non-linear charging relation of Pt in computational approach (would need reactivity DFT to handle oxide coverage as a function of potential); (2) controlling potential in DFT: theoretically impossible, as absolute potential does not exist; practically: need reference point or reference electrode – introducing further complications, but can be worked out… 
	
	o	what are the main attributes describing the catalyst activity and stability and how to compute them with a reasonable level of approximation?  
	o	system size exponentially scales the computational cost of DFT-based approaches 
	o	Moreover, addressing the complexity of space (particle size, shape, composition effects, etc.) specially at the interface, as well, statistical averaging over all possible configurations (surface vs bulk) at the electrolyte side is not feasible from first-principles methods alone
	
•	Explain how AI models can address the challenges explained above in DFT calculations [we should be more cautious: there will be problems for which this will work (based on DFT alone) and problems for which it will not work (based on DFT alone; at least not in the near future …) and require input from phenomenological theory and/or experiment; can we identify the problems/systems falling into these different categories? It is a dream to have one approach for all; but that might not be a realistic expectation for the foreseeable future…]

	o	Statistical averaging is inherent property of machine learning models
	o	Dimensionality reduction methods (e.g. principal component analysis) can be used for finding the descriptor for catalyst activity of a certain reaction 
	o	DFT-based neural network architecture along with monte-carlo simulations have been developed to simulate large scale systems (Artrith_2014)
	
•	Datasets (training, validation and test sets): needs standardized data to train model, typical problems in ML...

	o   De Luna_2017: Use machine learning to find energy materials	
	Cleaning process
	o	Size, classes, balance or imbalance (imbalance: challenging to train a classifier to predict)
	o	Split to test set and cross validation set for training and tuning hyperparameters
	o	Feature generation and labeling 
	o	ML method for classification
	o	Model training and validation 
	o	Model evaluation and benchmarking (test on independent datasets for performance evaluation)
	o	Feature extraction (similarity metrics: Euclidean distance, Manhattan distance, Pearson correlation score, KL-divergence)
	
•	An appropriate figure for this section: Formulate the approach with QM and machine learning (QM predictor based on machine learning) for simulation of electrode-electrolyte interface 

	o	Step 1: Sampling - Generate a combinatorial library; a “big data” using DFT to compute mixtures of materials (proper and relevant choices should be made here correspond to experiment, e.g. for various surface states, nanoparticles, bulk structures, slabs, with water layers, etc.; within an order of hundred thousand calculations needed (the larger the data the better to train the ML model). 
			Bottleneck to generate this library? human resources (efficient data harvesting methods needed), computational resources. The solution is in availability of a database system for materials design for a specific target reaction. VIM as will be discussed below. 
			Data should account for variables for the electronic structure of the solid electrode, solvent properties and ion distributions in the electrolyte as well as specific properties of a boundary region in-between. Idea: AIMD methods might be more efficiently employed for data generation (Groß_2014).  
	o	Train a machine learning method (e.g. tune the parameters of a neural net) for searching, classifying, or clustering the chemical space in terms of functionality -  this is an efficient screening/filtering of structure-property relationships to prioritize/predict the materials of interest (maybe only a hundred out of 100,000 are predicted as good) (Meredig_2014, Ward_2016), 
	o	use DFT again to calculate the properties of the predicted structures and evaluate the correlation between ML predicted and DFT calculated structures (Pyzer-Knapp_2015, Hachmann_2011). Filter out experimentally irrelevant structures.
	o	Finally, proposed material should be experimentally synthesized and be tested (essential step); Consequently, out of a hundred of predicted materials maybe only a few of them are useful.

## 2.	Review on recent AI models for materials simulation (to be completed by Mehrtoos and Mehrdad)

a.	learning the energy functional via examples (force field development)
i.	strategy, review of the works (Brockherde,_2017)
b.

•	Brief review: Generate a table of recent models for material simulation like below 

| First Header  | Second Header | Third Header  | Fourth Header | Fifth Header  | Sixth Header | Seventh Header  | Eighth Header |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| Content Cell  | Content Cell  | Content Cell  | Content Cell  | Content Cell  | Content Cell  | Content Cell  | Content Cell  |
| Content Cell  | Content Cell  | Content Cell  | Content Cell  | Content Cell  | Content Cell  | Content Cell  | Content Cell  |

Dataset (size, how generated)	Features/attributes 	Training algorithm	input	output	performance	Reference
DFT	Geometrical, electronic structure	Neural net	Atomic position	DFT energy	Compared with DFT	…
DFT	…	Autoencoder	Chemical structure	…	…	…
experiment	…	SVM	…	…	…	…
…	…	Genetic algorithm	…	…	…	…

•	Different flavors of ML have been used: graphical probabilistic models for reaction network (Ulissi_2017) 
•	as well as models based on empirical risk minimization (Goldsmith_2018) 
•	Neural network models (Yao_2018, Artrith_2014 Hy_2018)
•	Neural network potential-energy surfaces in chemistry: a tool for large-scale simulations (Behler_2011)
•	Generating latent space of a molecule using autoencoder and predictor. (Gómez-Bombarelli_2018)
•	More and more reviews needed here not only in electrocatalysis but also for other type of materials e.g. genetic algorithms extensively used for predicting macromolecules, drug design or polymers.

## 3.	Introducing Virtual Materials Intelligence Database
(To be completed by NRC team)

## 4.	Perspective for future research

## 5.	References

1. Aspuru-Guzik, Alan, Roland Lindh, and Markus Reiher. ACS central science 4.2 (2018): 144-152.
2. Duvenaud, David K., et al. Advances in neural information processing systems. 2015.
3. Jain, Anubhav, et al. "Commentary: The Materials Project: A materials genome approach to accelerating materials innovation." Apl Materials 1.1 (2013): 011002.
4. Agrawal, Ankit, and Alok Choudhary. Apl Materials 4.5 (2016): 053208.
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
