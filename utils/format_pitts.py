import os, csv, shutil
import numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd

from collections import defaultdict

class DatasetFormatter:
    def __init__(self, dataset_folder, output_folder):
        self.dataset_folder = os.path.join(dataset_folder)
        self.output_folder = output_folder
        self.outputs = ["Pittsburgh2A", "Pittsburgh2B", "Pittsburgh2C", "Pittsburgh2D"]
        self.inliers_threshold = 300
        self.load_filenames()
        self.extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()  # load the extractor
        self.matcher = LightGlue(features='superpoint').eval().cuda()  # load the matcher
    

    def load_filenames(self):
        self.image_paths = sorted(glob(os.path.join(self.dataset_folder, "**", "*.jpg"), recursive=True))
        self.image_infos = [self.parse_filename(path) for path in self.image_paths]


    def parse_filename(self, path):
        parts = path.split("@")
        info = {
            'utm_east': float(parts[1]),
            'utm_north': float(parts[2]),
            'city_id': 'Pittsburgh',
            'northdeg': '0',
            'lat': parts[5],
            'lon': parts[6],
            'panoid': 'Pittsburgh' + parts[7],
            'cameraid': parts[8],
            'timestamp': '199607', # unused
        }
        return info
    

    def image_matching(self, img0_path, img1_path):

        image0 = load_image(img0_path).cuda()
        image1 = load_image(img1_path).cuda()

        feats0 = self.extractor.extract(image0, resize = None)  # auto-resize the image, disable with resize=None
        feats1 = self.extractor.extract(image1, resize = None)

        matches01 = self.matcher({'image0': feats0, 'image1': feats1})

        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension

        kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
        m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

        return len(m_kpts0)

    def process_image_cvs(self, writer, index, place_id, output_folder, city_id):

        for sub_index in index:
            path = self.image_paths[sub_index]
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
                'panoid': info['panoid'] + info['cameraid'],
            })
            file_name = f"{str(city_id)}_{str(place_id).zfill(7)}_{year}_{month}_{str(info['northdeg']).zfill(3)}_{float(info['lat'])}_{float(info['lon'])}_{info['panoid'] + info['cameraid']}.jpg"
            shutil.copyfile(path, os.path.join(output_folder, file_name))


    def absoulte_positive(self, candidate, queries, record):

        refine_class_index = []
        matches_sum = {}

        query_image_path = self.image_paths[queries]

        for sub_index in candidate:
            if sub_index not in record:
                image_path = self.image_paths[sub_index]
                num_inliers = self.image_matching(query_image_path, image_path)
                matches_sum[sub_index] = num_inliers

        sorted_matches = sorted(matches_sum.items(), key=lambda x: x[1], reverse=True)

        refine_class_index.append(queries)
        for sub_index, num_inliers in sorted_matches:
            if num_inliers > self.inliers_threshold:
                record.append(sub_index)
                refine_class_index.append(sub_index)

        return refine_class_index


    def to_gsv_format(self):
        
        class_id__group_id = [DatasetFormatter.get__class_id__group_id(info['utm_east'], info['utm_north'])
                              for info in self.image_infos]
        
        images_per_class = defaultdict(list)
        for i, (class_id, _) in enumerate(class_id__group_id):
            images_per_class[class_id].append(i)
        
        images_per_class = {k: v for k, v in images_per_class.items() if len(v) >= 25}
        
        classes_per_group = defaultdict(set)
        for class_id, group_id in class_id__group_id:
            if class_id not in images_per_class:
                continue  # Skip classes with too few images
            classes_per_group[group_id].add(class_id)
        
        classes_per_group = [list(c) for c in classes_per_group.values()]


        for i, group_id in enumerate(classes_per_group):

            sub_output_folder = os.path.join(self.output_folder, "Images", self.outputs[i])
            if not os.path.exists(sub_output_folder):
                os.makedirs(sub_output_folder)
            output_csv = os.path.join(self.output_folder, "Dataframes", self.outputs[i] + ".csv")
            place_id = 0

            with open(output_csv, mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=['place_id', 'year', 'month', 'northdeg', 'city_id', 'lat', 'lon', 'panoid'])
                writer.writeheader()

                for class_id in tqdm(group_id):
                    for j, images_id in enumerate(images_per_class[class_id]):
                        record_images_id = []
                        if j in [0, 3, 6, 9]:
                            record_images_id.append(images_id)
                            refine_class_index = self.absoulte_positive(images_per_class[class_id][j+24:], images_id, record_images_id)
                            if len(refine_class_index) >= 4:
                                self.process_image_cvs(writer, refine_class_index, place_id, sub_output_folder, self.outputs[i])
                                place_id+=1

    @staticmethod
    def get__class_id__group_id(utm_east, utm_north, M = 25, N = 2):
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


csv_generator = DatasetFormatter('/mnt/sda3/Projects/VPR-datasets-downloader-main/datasets/pitts250k/images/train', 
                                 '/mnt/sda3/Projects/npr/datasets/gsv_cities')
csv_generator.to_gsv_format()