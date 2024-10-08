# https://github.com/amaralibey/gsv-cities

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

IMAGENET_MEAN_STD = {'mean': [0.485, 0.456, 0.406],
                     'std': [0.229, 0.224, 0.225]}

# GSV-Cities
TRAIN_CITIES = ['Bangkok', 'BuenosAires', 'LosAngeles', 'MexicoCity', 'OSL',
                'Rome', 'Barcelona', 'Chicago', 'Madrid', 'Miami', 'Phoenix',
                'TRT', 'Boston', 'Lisbon', 'Medellin', 'Minneapolis',
                'PRG', 'WashingtonDC', 'Brussels', 'London',
                'Melbourne', 'Osaka', 'PRS',]

EXTRA_DATASETS_MSLS = ['MSLS0austin', 'MSLS0bangkok', 'MSLS0berlin', 'MSLS0boston', 'MSLS0budapest', 'MSLS0helsinki', 'MSLS0melbourne', 
               'MSLS0moscow', 'MSLS0ottawa', 'MSLS0paris', 'MSLS0phoenix', 'MSLS0saopaulo', 'MSLS0tokyo', 'MSLS0toronto', 
               'MSLS10austin', 'MSLS10bangkok', 'MSLS10berlin', 'MSLS10boston', 'MSLS10budapest', 'MSLS10helsinki', 
               'MSLS10melbourne', 'MSLS10moscow', 'MSLS10ottawa', 'MSLS10paris', 'MSLS10phoenix', 'MSLS10saopaulo', 'MSLS10tokyo', 
               'MSLS11austin', 'MSLS11bangkok', 'MSLS11berlin', 'MSLS11boston', 'MSLS11budapest', 'MSLS11helsinki', 'MSLS11melbourne', 
               'MSLS11moscow', 'MSLS11ottawa', 'MSLS11paris', 'MSLS11phoenix', 'MSLS11saopaulo', 'MSLS11tokyo', 
               'MSLS12austin', 'MSLS12bangkok', 'MSLS12berlin', 'MSLS12boston', 'MSLS12budapest', 'MSLS12helsinki', 
               'MSLS12melbourne', 'MSLS12moscow', 'MSLS12ottawa', 'MSLS12paris', 'MSLS12phoenix', 'MSLS12saopaulo', 'MSLS12tokyo', 
               'MSLS13austin', 'MSLS13bangkok', 'MSLS13berlin', 'MSLS13boston', 'MSLS13budapest', 'MSLS13helsinki', 'MSLS13melbourne', 
               'MSLS13moscow', 'MSLS13ottawa', 'MSLS13paris', 'MSLS13phoenix', 'MSLS13saopaulo', 'MSLS13tokyo', 
               'MSLS14austin', 'MSLS14bangkok', 'MSLS14berlin', 'MSLS14boston', 'MSLS14budapest', 'MSLS14helsinki', 
               'MSLS14melbourne', 'MSLS14moscow', 'MSLS14ottawa', 'MSLS14paris', 'MSLS14phoenix', 'MSLS14saopaulo', 
               'MSLS14tokyo', 'MSLS14toronto', 'MSLS15austin', 'MSLS15bangkok', 'MSLS15berlin', 'MSLS15boston', 'MSLS15budapest', 
               'MSLS15helsinki', 'MSLS15melbourne', 'MSLS15moscow', 'MSLS15ottawa', 'MSLS15paris', 'MSLS15phoenix', 'MSLS15saopaulo', 'MSLS15tokyo', 'MSLS15toronto', 
               'MSLS16austin', 'MSLS16bangkok', 'MSLS16berlin', 'MSLS16boston', 'MSLS16budapest', 'MSLS16helsinki', 
               'MSLS16melbourne', 'MSLS16moscow', 'MSLS16ottawa', 'MSLS16paris', 'MSLS16phoenix', 'MSLS16saopaulo', 'MSLS16tokyo', 'MSLS16toronto', 
               'MSLS17austin', 'MSLS17bangkok', 'MSLS17berlin', 'MSLS17boston', 'MSLS17budapest', 'MSLS17helsinki', 'MSLS17melbourne', 'MSLS17moscow', 
               'MSLS17ottawa', 'MSLS17paris', 'MSLS17phoenix', 'MSLS17saopaulo', 'MSLS17tokyo', 
               'MSLS18austin', 'MSLS18bangkok', 'MSLS18berlin', 'MSLS18boston', 'MSLS18budapest', 'MSLS18helsinki', 'MSLS18melbourne', 
               'MSLS18moscow', 'MSLS18ottawa', 'MSLS18paris', 'MSLS18phoenix', 'MSLS18saopaulo', 'MSLS18tokyo', 
               'MSLS19austin', 'MSLS19bangkok', 'MSLS19berlin', 'MSLS19boston', 'MSLS19budapest', 'MSLS19helsinki', 'MSLS19melbourne', 
               'MSLS19moscow', 'MSLS19ottawa', 'MSLS19paris', 'MSLS19phoenix', 'MSLS19saopaulo', 'MSLS19tokyo', 'MSLS1austin', 
               'MSLS1bangkok', 'MSLS1berlin', 'MSLS1boston', 'MSLS1budapest', 'MSLS1helsinki', 'MSLS1melbourne', 'MSLS1moscow', 
               'MSLS1ottawa', 'MSLS1paris', 'MSLS1phoenix', 'MSLS1saopaulo', 'MSLS1tokyo', 'MSLS1toronto', 
               'MSLS20austin', 'MSLS20bangkok', 'MSLS20berlin', 'MSLS20boston', 'MSLS20budapest', 'MSLS20helsinki', 'MSLS20melbourne', 
               'MSLS20moscow', 'MSLS20ottawa', 'MSLS20paris', 'MSLS20phoenix', 'MSLS20saopaulo', 'MSLS20tokyo', 'MSLS20toronto', 
               'MSLS21austin', 'MSLS21bangkok', 'MSLS21berlin', 'MSLS21boston', 'MSLS21budapest', 'MSLS21helsinki', 'MSLS21melbourne', 
               'MSLS21moscow', 'MSLS21ottawa', 'MSLS21paris', 'MSLS21phoenix', 'MSLS21saopaulo', 'MSLS21tokyo', 'MSLS22austin', 
               'MSLS22bangkok', 'MSLS22berlin', 'MSLS22boston', 'MSLS22budapest', 'MSLS22helsinki', 'MSLS22melbourne', 'MSLS22moscow', 
               'MSLS22ottawa', 'MSLS22paris', 'MSLS22phoenix', 'MSLS22saopaulo', 'MSLS22tokyo', 
               'MSLS23austin', 'MSLS23bangkok', 'MSLS23berlin', 'MSLS23boston', 'MSLS23budapest', 'MSLS23helsinki', 'MSLS23melbourne', 
               'MSLS23moscow', 'MSLS23ottawa', 'MSLS23paris', 'MSLS23phoenix', 'MSLS23saopaulo', 'MSLS23tokyo', 'MSLS23toronto', 
               'MSLS24austin', 'MSLS24bangkok', 'MSLS24berlin', 'MSLS24boston', 'MSLS24budapest', 'MSLS24helsinki', 'MSLS24melbourne', 
               'MSLS24moscow', 'MSLS24ottawa', 'MSLS24paris', 'MSLS24phoenix', 'MSLS24saopaulo', 'MSLS24tokyo', 
               'MSLS2austin', 'MSLS2bangkok', 'MSLS2berlin', 'MSLS2boston', 'MSLS2budapest', 'MSLS2helsinki', 'MSLS2melbourne', 'MSLS2moscow', 
               'MSLS2ottawa', 'MSLS2paris', 'MSLS2phoenix', 'MSLS2saopaulo', 'MSLS2tokyo', 'MSLS2toronto', 'MSLS3austin', 'MSLS3bangkok', 
               'MSLS3berlin', 'MSLS3boston', 'MSLS3budapest', 'MSLS3helsinki', 'MSLS3melbourne', 'MSLS3moscow', 'MSLS3ottawa', 'MSLS3paris', 
               'MSLS3phoenix', 'MSLS3saopaulo', 'MSLS3tokyo', 'MSLS3toronto', 
               'MSLS4austin', 'MSLS4bangkok', 'MSLS4berlin', 'MSLS4boston', 'MSLS4budapest', 'MSLS4helsinki', 'MSLS4melbourne', 'MSLS4moscow', 
               'MSLS4ottawa', 'MSLS4paris', 'MSLS4phoenix', 'MSLS4saopaulo', 'MSLS4tokyo', 'MSLS4toronto', 
               'MSLS5austin', 'MSLS5bangkok', 'MSLS5berlin', 'MSLS5boston', 'MSLS5budapest', 'MSLS5helsinki', 'MSLS5melbourne', 
               'MSLS5moscow', 'MSLS5ottawa', 'MSLS5paris', 'MSLS5phoenix', 'MSLS5saopaulo', 'MSLS5tokyo', 
               'MSLS6austin', 'MSLS6bangkok', 'MSLS6berlin', 'MSLS6boston', 'MSLS6budapest', 'MSLS6helsinki', 'MSLS6melbourne', 'MSLS6moscow', 
               'MSLS6ottawa', 'MSLS6paris', 'MSLS6phoenix', 'MSLS6saopaulo', 'MSLS6tokyo', 
               'MSLS7austin', 'MSLS7bangkok', 'MSLS7berlin', 'MSLS7boston', 'MSLS7budapest', 'MSLS7helsinki', 'MSLS7melbourne', 'MSLS7moscow', 
               'MSLS7ottawa', 'MSLS7paris', 'MSLS7phoenix', 'MSLS7saopaulo', 'MSLS7tokyo', 'MSLS7toronto', 
               'MSLS8austin', 'MSLS8bangkok', 'MSLS8berlin', 'MSLS8boston', 'MSLS8budapest', 'MSLS8helsinki', 'MSLS8melbourne', 'MSLS8moscow', 
               'MSLS8ottawa', 'MSLS8paris', 'MSLS8phoenix', 'MSLS8saopaulo', 'MSLS8tokyo', 
               'MSLS9austin', 'MSLS9bangkok', 'MSLS9berlin', 'MSLS9boston', 'MSLS9budapest', 'MSLS9helsinki', 'MSLS9melbourne', 'MSLS9moscow', 'MSLS9ottawa', 
               'MSLS9paris', 'MSLS9phoenix', 'MSLS9saopaulo', 'MSLS9tokyo', 'MSLS9toronto', 'MSLS9trondheim']


EXTRA_DATASETS_SF = ["SFXL0", "SFXL1", "SFXL2"]

EXTRA_DATASETS_PITTS = ['Pittsburgh2A','Pittsburgh2B','Pittsburgh2C','Pittsburgh2D',]

COMPARISON_FOR_CM = EXTRA_DATASETS_MSLS + TRAIN_CITIES

EXTRA_DATASETS = EXTRA_DATASETS_PITTS + EXTRA_DATASETS_SF + EXTRA_DATASETS_MSLS

GPMS_DATASETS = EXTRA_DATASETS + TRAIN_CITIES

class GSVCitiesDataset(Dataset):
    def __init__(self, args, cities=['London', 'Boston'], img_per_place=4, min_img_per_place=4):
        super(GSVCitiesDataset, self).__init__()
        self.base_path = os.path.join(args.datasets_folder, "gsv_cities/")
        self.cities = cities
        self.is_inference = False

        assert img_per_place <= min_img_per_place, f"img_per_place should be less than {min_img_per_place}"
        self.img_per_place = img_per_place
        self.min_img_per_place = min_img_per_place
        self.transform = transforms.Compose([transforms.Resize(args.resize, interpolation=transforms.InterpolationMode.BILINEAR),
                                    transforms.RandAugment(num_ops=3, interpolation=transforms.InterpolationMode.BILINEAR),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=IMAGENET_MEAN_STD['mean'], std=IMAGENET_MEAN_STD['std']),])

        # generate the dataframe contraining images metadata
        self.dataframe = self.__getdataframes()

        # get all unique place ids
        self.places_ids = pd.unique(self.dataframe.index)
        self.total_nb_images = len(self.dataframe)

    def __getdataframes(self):
        ''' 
            Return one dataframe containing
            all info about the images from all cities

            This requieres DataFrame files to be in a folder
            named Dataframes, containing a DataFrame
            for each city in self.cities
        '''
        # read the first city dataframe
        df = pd.read_csv(self.base_path+'Dataframes/'+f'{self.cities[0]}.csv')
        df = df.sample(frac=1)  # shuffle the city dataframe

        # append other cities one by one
        for i in range(1, len(self.cities)):
            tmp_df = pd.read_csv(self.base_path+'Dataframes/'+f'{self.cities[i]}.csv')

            prefix = i
            tmp_df['place_id'] = tmp_df['place_id'] + (prefix * 10**5)
            tmp_df = tmp_df.sample(frac=1)  # shuffle the city dataframe

            df = pd.concat([df, tmp_df], ignore_index=True)

        # keep only places depicted by at least min_img_per_place images
        res = df[df.groupby('place_id')['place_id'].transform(
            'size') >= self.min_img_per_place]
        return res.set_index('place_id')

    def __getitem__(self, index):

        if self.is_inference:
            place_id = self.places_ids[index]
            row = self.dataframe.loc[place_id].iloc[0]
            img_name = self.get_img_name(row)
            img_path = self.base_path + 'Images/' + \
                row['city_id'] + '/' + img_name
            img = self.image_loader(img_path)
            img = self.transform(img)

            return img, torch.tensor([place_id])


        place_id = self.places_ids[index]

        # get the place in form of a dataframe (each row corresponds to one image)
        place = self.dataframe.loc[place_id]
        place = place.sample(n=self.img_per_place)
        imgs = []
        for i, row in place.iterrows():
            img_name = self.get_img_name(row)
            img_path = self.base_path + 'Images/' + \
                row['city_id'] + '/' + img_name
            img = self.image_loader(img_path)

            img = self.transform(img)

            imgs.append(img)

        # [BS, K, channels, height, width])
        return torch.stack(imgs), torch.tensor(place_id).repeat(self.img_per_place)

    def __len__(self):
        '''Denotes the total number of places (not images)'''
        return len(self.places_ids)

    @staticmethod
    def image_loader(path):
        return Image.open(path).convert('RGB')

    @staticmethod
    def get_img_name(row):
        # given a row from the dataframe
        # return the corresponding image name

        city = row['city_id']

        # now remove the two digit we added to the id
        # they are superficially added to make ids different
        # for different cities
        # row.name is the index of the row, not to be confused with image name
        pl_id = row.name % 10**5
        pl_id = str(pl_id).zfill(7)

        panoid = row['panoid']
        year = str(row['year']).zfill(4)
        month = str(row['month']).zfill(2)
        northdeg = str(row['northdeg']).zfill(3)
        lat, lon = str(row['lat']), str(row['lon'])
        name = city+'_'+pl_id+'_'+year+'_'+month+'_' + \
            northdeg+'_'+lat+'_'+lon+'_'+panoid+'.jpg'
        return name
