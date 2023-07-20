## Paper 1: Robot Team II Electric Boogaloo
### Previous outline: 
1. Introduction
   - Human Impact / Importance 
   - Hyperspectral Imaging and Remote sensing
     - Satellites with HSI 
     - Drones with HSI 
     - applications 
       - Spectral Indices + Vegetation Studies
       - Machine Learning with HSI
   - Remote Sensing applications to characterization of bodies of water
     - meteorology 
     - atmospheric sensing 
     - need for in-istu sensing
   - Autonomous Robotic Teams 
     - coordination
     - autonomous data collection 
2. Materials & Methods 
   - Robotic Team 
   - Fluorometers (technology and our included sensors)
   - Data collection procedure
     - AV -> Boat -> AV -> Maps -> Verification w/ Boat
   - Software 
     - Julia & MLJ
     - Docker + Influxdb + NodeRed + Grafana for live visualizations
   - Processing of HSIs 
     - radiance + downwelling irradiance -> reflectances 
       - assumption of *Lambertian* surface
     - georectification 
       - georectification = orthorectify + georeference
     - Other important variables 
       - Viewing geometry (roll, pitch, yaw)
       - Solar geometry i.e. incident geometry (solar azimuth, solar elevation)
         - should visualize this... see the [wikipedia image](https://en.wikipedia.org/wiki/Solar_zenith_angle#/media/File:Solar_Zenith_Angle_min.png)
       - mention grazing incidence reflection and ideal time for data collection
     - Data Collocation procedure 
       - `ilat`, `ilon` and representativeness uncertainty
     - Machine Learning Methods 
       - Individual learners 
       - Feature importance ranking 
         - discuss pros/cons and mentions that Shapley values take too long to compute to be helpful here 
       - Hyperparameter optimization
       - Super learner / Model stacking 
         - discuss training on complimentary folds
       - MLJ model search and data pipeline (i.e. learning networks and DAGs)
         - we should make a graph to visualize the data pipeline for the superlearner stack 
3. Results 
   - Exploratory Data Analysis results 
   - Combining multiple days of data (i.e. the difference between this and the previous paper)
   - Comparison of vanilla models, hpo models and super learner
   - Updated maps of Scotty's ranch 
4. Discussion

