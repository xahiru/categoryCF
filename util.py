from dataset import Dataset
import pandas as pd


file_path_save_data = 'data/processed/' 
datasetname = 'ml-100k'  # valid datasetnames are 'ml-latest-small', 'ml-20m', and 'jester'
data1 = Dataset.load_builtin(datasetname)
       
path = '~/.surprise_data/ml-100k/ml-100k/u.item'
df = pd.read_csv(path, sep="|", encoding="iso-8859-1", names=['id','name','date','space','url','cat1','cat2','cat3','cat4','cat5','cat6','cat7','cat8','cat9','cat10','cat11','cat12','cat13','cat14','cat15','cat16','cat17','cat18','cat19'])
list_of_cats = {}

df1 = df[['id','cat1','cat2','cat3','cat4','cat5','cat6','cat7','cat8','cat9','cat10','cat11','cat12','cat13','cat14','cat15','cat16','cat17','cat18','cat19']]
for row in df.itertuples(index=True, name='Pandas'):
    id = str(getattr(row, "id"))
    cate_x = [getattr(row, "cat1"),getattr(row, "cat2"),getattr(row, "cat3"),getattr(row, "cat4"),getattr(row, "cat5"),getattr(row, "cat6"),getattr(row, "cat7"),getattr(row, "cat8"),getattr(row, "cat9"),getattr(row, "cat10"),getattr(row, "cat11"),getattr(row, "cat12"),getattr(row, "cat13"),getattr(row, "cat14"),getattr(row, "cat15"),getattr(row, "cat16"),getattr(row, "cat17"),getattr(row, "cat18"),getattr(row, "cat19"),]
    list_of_cats[id] = cate_x



# print(df1)
# df1.to_csv(file_path_save_data+'itemswithcat.csv', sep='\t', encoding='utf-8')
# print(list_of_cats)