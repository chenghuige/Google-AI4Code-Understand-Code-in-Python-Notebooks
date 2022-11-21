rsync -avP --delete pikachu/*.json kaggle
rsync -avP --delete pikachu/third kaggle
rsync -avP --delete pikachu/utils kaggle
#kaggle datasets create -p kaggle --dir-mode zip
kaggle datasets version -p kaggle --dir-mode zip -m "m"
