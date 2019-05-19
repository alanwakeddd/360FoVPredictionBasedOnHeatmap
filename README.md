# 360FoVPredictionBasedOnHeatmap
Project Team 1 of ECE6123- Image and Video Processing (Spring 2019) – NYU

        
        Team Member:

        Tianfei Song (N)
        Meijuan Wang (N15457318)
        Bohan Zhang (N13992422)
<br />
We treat FOV prediction as a sequence to sequence problem. Using heatmap of 10 secs as the input, then predicting the 
heatmap of future ten secs.
</ br>
we use the dataset of Shanghai to make the prediction. For Convlstm model we use the code from Chenge Li.
###
https://github.com/ChengeLi/360FoV
<br />
Use shanghai factory method in data_provider/npz_builder.py to convert the Shanghai dataset to certain input type for predrnn++.
You may need to change paramenters for last several lines of npz_builder.py to deal with your own dataset.
You may need to change configure paramenters for train.py to train this model on your own computer.
You can find complete predrnn++ code for this project in predrnn-pp-master.rar.
Original version of predrnn++ is listed below.
<br />
###
https://github.com/Yunbo426/predrnn-pp
