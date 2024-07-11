import pandas as pd
import numpy as np
import utils # personal utility functions

class Processor():
    def __init__(self, data: pd.DataFrame = None) -> None:
        self.__data = data
        # if not {"longitude", "latitude"}.issubset(data.columns):
        #     # Will make the following line into error warning to interrupt the program.
        #     print("Should provide files with 'longitude' and 'latitude' included!")

    def coord_twd97towgs84(self) -> None:
        self.__data["lon_new"], self.__data["lat_new"] = utils.twd97_to_lonlat_srs(
            twd97_x = self.__data["longitude"],
            twd97_y = self.__data["latitude"]
        )
        return

    def pt_vs_srs_dist(self, pt: list = [121, 25], srs: pd.DataFrame = None) -> pd.Series:
        """
        TBD.

        Parameters
        ----------
        pt : list (but any list-like container would work)
            Containing longitude-latitude coordinate. Would be the anchor point.
        srs : pd.DataFrame
            List of coordinates to calculate distances with the single point.

        Returns
        -------
        pd.Series: calculated series of distance from the single point
        """
        R = 6371

        srs["X"] = np.radians(srs["_X"])
        srs["Y"] = np.radians(srs["_Y"])
        pt = np.radians(pt)

        srs["dlon"] = pt[0] - srs["X"]
        srs["dlat"] = pt[1] - srs["Y"]
        srs["a"] = (np.sin(srs["dlat"]/2) ** 2) + np.cos(srs["X"]) * np.cos(pt[0]) * (np.sin(srs["dlon"]/2) ** 2)

        srs["dist"] = 2 * np.arcsin(np.sqrt(srs["a"])) * R * 1000

        return list(srs[["_X", "_Y", "index"]].loc[srs["dist"].idxmin()])

    def find_nearest_point(self, filepath: str = ".\\Rental_geocode\\", ref_file_ver: int = None) -> None:
        """
        TBD.

        Parameters
        ----------
        filepath : str. Default = ".\\Rental_geocode\\".
            Path to these policy files.
        ref_file_ver : int. Default = None.
            The version of policy files, currently accepting 1 to 4.

        Returns
        -------
        None
        """
        if not ref_file_ver or ref_file_ver < 1 or ref_file_ver > 4:
            print("Please provide a valid policy segment file for referencing")
            return
        
        policy_file = f"policy{ref_file_ver}_segments.xlsx"
        print(f"Processing {policy_file[:-5]}...")
        
        policy_file = filepath + policy_file
        suffixes = {
            1: "201006",
            2: "201012",
            3: "201406",
            4: "201508"
        }

        policy_points = pd.read_excel(policy_file)[['_X', '_Y', 'index']]

        # if one of the long and lat is NaN, directly set the value to NaN.

        if 'merged_coord' not in self.__data.columns:
            # TODO: this does not work as I thought, so I use a more stupid way below
            self.__data['merged_coord'] = np.where(
                self.__data.isna().any(axis=1),
                np.nan,
                list(zip(self.__data['lon_new'], self.__data['lat_new']))
            )

            # tmp = list(zip(self.__data['lon_new'], self.__data['lat_new']))
            # self.__data['merged_coord'] = tmp


        # self.__data['results'] = self.__data['merged_coord'].map(lambda x: self.pt_vs_srs_dist(pt=x, srs=policy_points))

        # TODO: delete after the code works
        # result = list(map(lambda x: add_and_multiply(x, y, z), numbers))
        # squared = data.map(lambda x: x ** 2)

        # self.__data[f'seg_lon_{suffixes[ref_file_ver]}'] = ""
        # self.__data[f'seg_lat_{suffixes[ref_file_ver]}'] = ""
        # self.__data[f'segment_{suffixes[ref_file_ver]}'] = ""

        return

    def show_data(self, topX: int = None, is_thin: bool = False) -> None:
        data_to_show = self.__data

        if is_thin:
            thin_list = ['address', 'longitude', 'latitude', 'lon_new', 'lat_new']
            thin_list = ['lon_new', 'lat_new', 'merged_coord']
            data_to_show = data_to_show[thin_list]

        if topX:
            print(data_to_show.head(topX))
        else:
            print(data_to_show)
        return

    def get_data(self) -> pd.DataFrame:
        pass

    def output_data(self, format: str = "csv") -> None:
        '''
        Can output data to csv, excel, or Stata dta
        '''
        pass


def main():
    filepath = ".\\Rental_geocode\\不動產租賃_master.dta"

    p = Processor(pd.read_stata(filepath))
    # p = Processor()
    p.coord_twd97towgs84()
    for i in range(1):
        p.find_nearest_point(ref_file_ver=i+1)
    
    p.show_data(is_thin=True)



if __name__ == "__main__":
    main()