## Requirements

```
conda env create -f InteractNet.yml
conda activate noveldti
```
## Data
All data used in this article is available here: Human dataset：https://github.com/masashitsubaki/CPI_prediction/tree/master/dataset, DUD-E：http://dude.docking.org ,
 BindingDB：https://github.com/IBM/InterpretableDTIP,  Human dataset：https://github.com/masashitsubaki/CPI_prediction/tree/master/dataset,  and human sequence to pdb：https://github.com/prokia/drugVQA/tree/master/data, Yamanishi_08’s dataset ： https://drugtargets.insight-centre.org/.

## Demo
After downloading the human dataset you and place it in the project root folder you can generate the preprocessed data by running
```
python human_data.py
```
After generating the human_part_train.pkl, human_part_val.pkl and human_part_test.pkl you can start training the model by running
```
python main.py
```





