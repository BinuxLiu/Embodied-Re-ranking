import os, csv, shutil
import numpy as np
from glob import glob
from tqdm import tqdm
from collections import defaultdict

class DatasetFormatter:
    def __init__(self, dataset_folder, output_folder):
        self.dataset_folder = dataset_folder
        self.name = "SFXL2"
        self.output_folder = os.path.join(output_folder, "Images", self.name)
        self.output_csv = os.path.join(output_folder, "Dataframes", self.name + ".csv")
        self.load_filenames()

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
    
    def load_filenames(self):
        self.image_paths = sorted(glob(os.path.join(self.dataset_folder, "**", "*.jpg"), recursive=True))
        self.image_infos = [self.parse_filename(path) for path in self.image_paths]

    def parse_filename(self, path):
        parts = path.split("@")
        info = {
            'utm_east': float(parts[1]),
            'utm_north': float(parts[2]),
            'city_id': self.name,
            'northdeg': parts[9],
            'lat': parts[5],
            'lon': parts[6],
            'panoid': parts[7],
            'timestamp': parts[13],
        }
        return info
    
    def new_path(self, path, level_name = "SFXL1"):

        parts = path.split('/')
        parts[7] = "processed"
        file_name = parts[-1]
        if level_name == "SFXL1":
            modified_file_name = file_name.replace("@0@", "@120@", 1)
        else:
            modified_file_name = file_name.replace("@0@", "@240@", 1)
        parts[-1] = modified_file_name
        modified_path = '/'.join(parts)

        return modified_path

    def process_image(self, writer, index, place_id, output_folder, city_id):

        for sub_index in index:
            path = self.image_paths[sub_index]
            if self.name in ["SFXL2", "SFXL1"]:
                path = self.new_path(path, self.name)
            info = self.image_infos[sub_index]
            timestamp = info['timestamp']
            year, month = timestamp[:4], timestamp[4:6]
            writer.writerow({
                'place_id': str(place_id).zfill(4),
                'year': year,
                'month': month,
                'northdeg': info['northdeg'],
                'city_id': city_id,
                'lat': info['lat'],
                'lon': info['lon'],
                'panoid': info['panoid'],
            })
            file_name = f"{str(city_id)}_{str(place_id).zfill(7)}_{year}_{month}_{str(info['northdeg']).zfill(3)}_{float(info['lat'])}_{float(info['lon'])}_{info['panoid']}.jpg"
            shutil.copyfile(path, os.path.join(output_folder, file_name))

    def to_gsv_format(self):
            
        class_id__group_id = [DatasetFormatter.get__class_id__group_id(info['utm_east'], info['utm_north'])
                              for info in self.image_infos]
        
        images_per_class = defaultdict(list)
        for i, (class_id, _) in enumerate(class_id__group_id):
            images_per_class[class_id].append(i)
        
        images_per_class = {k: v for k, v in images_per_class.items() if len(v) >= 10}
        
        classes_per_group = defaultdict(set)
        for class_id, group_id in class_id__group_id:
            if class_id not in images_per_class:
                continue  # Skip classes with too few images
            classes_per_group[group_id].add(class_id)
        
        classes_per_group = [list(c) for c in classes_per_group.values()]

        with open(self.output_csv, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['place_id', 'year', 'month', 'northdeg', 'city_id', 'lat', 'lon', 'panoid'])
            writer.writeheader()

            place_id = 0
            for class_id in tqdm(classes_per_group[0]):
                self.process_image(writer, images_per_class[class_id], place_id, self.output_folder, self.name)
                place_id+=1

    @staticmethod
    def get__class_id__group_id(utm_east, utm_north, M = 10, N = 5):
        """Return class_id and group_id for a given point.
            The class_id is a triplet (tuple) of UTM_east, UTM_north and
            heading (e.g. (396520, 4983800,120)).
            The group_id represents the group to which the class belongs
            (e.g. (0, 1, 0)), and it is between (0, 0, 0) and (N, N, L).
        """
        rounded_utm_east = int(utm_east // M * M)  # Rounded to nearest lower multiple of M
        rounded_utm_north = int(utm_north // M * M)
        
        class_id = (rounded_utm_east, rounded_utm_north)
        # group_id goes from (0, 0) to (N, N)
        group_id = (rounded_utm_east % (M * N) // M,
                    rounded_utm_north % (M * N) // M)
        return class_id, group_id

def main():
    csv_generator = DatasetFormatter('/mnt/sda3/Projects/npr/datasets/sf_xl/small/train',
                                     '/mnt/sda3/Projects/npr/datasets/gsv_cities')
    csv_generator.to_gsv_format()


if __name__ == '__main__':
    main()