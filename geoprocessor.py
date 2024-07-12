import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import datetime
import utils # personal utility functions

# NOTE: visualize progress of pandas operations
# the params can be placed here for pandas tqdm display.
# tqdm.pandas()

# TODO:Currently the policy version numbers are scattered in functions
# might want to set them as parameters to pass into GeoProcessor

class GeoProcessor():
    def __init__(self, data: pd.DataFrame = None) -> None:
        self.__data = data
        if not {"longitude", "latitude"}.issubset(data.columns):
            # TODO: make the following line into error warning to interrupt the program.
            print("Should provide files with 'longitude' and 'latitude' included!")

    def _make_shorter(self) -> None:
        '''
        This method is to make the dataframe shorter for testing.
        Will extract only top and bottom 10 observations, but will keep all
        columns.
        '''
        # NOTE: np.r_ for multiple index range slicing
        self.__data = self.__data.iloc[np.r_[:10, -11:-1]].reset_index(drop=True)

    def coord_twd97towgs84(self) -> None:
        self.__data["lon_new"], self.__data["lat_new"] = utils.twd97_to_lonlat_srs(
            twd97_x = self.__data["longitude"],
            twd97_y = self.__data["latitude"]
        )
        return

    def pt_vs_srs_dist(self, pt: list = [121, 25], srs: pd.DataFrame = None) -> pd.Series:
        """
        Since the purpose of the main program is to find the shortest distance,
        whether using the distance on a sphere or actual Earth does not matter.
        Therefore, the formula to calculate distance on spheres is used here,
        for I haven't found a vectorized way to calculate the actual geo distance. 

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

        return list(srs[["_X", "_Y", "dist", "index"]].loc[srs["dist"].idxmin()])

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
        # TODO: policy version control.
        if not ref_file_ver or ref_file_ver < 1 or ref_file_ver > 4:
            print("Please provide a valid policy segment file for referencing")
            return
        
        policy_file = f"policy{ref_file_ver}_segments.xlsx"
        print(f"\n>> Processing {policy_file[:-5]}...")
        
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
            # NOTE: this does not work as I thought, so I use a more intuitive way below
            # self.__data['merged_coord'] = np.where(
            #     self.__data.isna().any(axis=1),
            #     np.nan,
            #     list(zip(self.__data['lon_new'], self.__data['lat_new']))
            # )
            self.__data['merged_coord'] = [
                np.nan if np.isnan(lon) or np.isnan(lat) else (lon, lat) 
                for (lon, lat) in tqdm(
                    zip(self.__data['lon_new'], self.__data['lat_new']), 
                    desc = "[Preprocess] Merging coords",
                    total = self.__data.shape[0]
                    )
            ]

        # progress_map works the same as pandas map, only works after using 
        # tqdm, used to track progress.
        tqdm.pandas(
            desc = "Finding nearest pt",
            total = self.__data['merged_coord'].count()
        )
        self.__data['results'] = self.__data['merged_coord'] \
            .progress_map(lambda x: self.pt_vs_srs_dist(pt=x, srs=policy_points), na_action="ignore")

        # split the results to three new columns, need to first replace all np.nan
        # with [np.nan, np.nan, np.nan], since splitting does not ignore np.nan, worth doing 
        # some research on this tho.
        new_cols = [
            f'seg_lon_{suffixes[ref_file_ver]}', 
            f'seg_lat_{suffixes[ref_file_ver]}',
            f'distance_{suffixes[ref_file_ver]}',
            f'segment_{suffixes[ref_file_ver]}'
        ]

        tmp_n = len(new_cols)
        self.__data['results'] = self.__data['results'].map(lambda x: x if isinstance(x, list) else [np.nan]*tmp_n)
        
        self.__data[new_cols] = pd.DataFrame(self.__data['results'].tolist())
        self.__data.drop(columns=['results'], inplace=True)

        # TODO: policy version control.
        if ref_file_ver == 4: # last policy file done processing
            self.__data.drop(columns=['merged_coord'], inplace=True)
        
        return

    def show_data(self, topX: int = None, is_thin: bool = False) -> None:
        list_to_show = self.__data.columns

        if is_thin:
            list_to_show = ['address', 'longitude', 'latitude', 'lon_new', 'lat_new']

        if topX:
            print(self.__data[list_to_show].head(topX))
        else:
            print(self.__data[list_to_show])
        return

    def get_data(self) -> pd.DataFrame:
        return self.__data

    def output_data(
            self, 
            format: str = "csv", 
            new_name: str = "rental_master", 
            want_sep: bool = False
        ) -> None:
        '''
        Can output data to csv, excel, or Stata dta.
        '''
        print("\nStart data output......", end="")
        format_l = format.lower()
        columns = self.__data.columns
        max_col_cnt = len(columns)

        def output(cols, new_name):
            if format_l == "csv":
                self.__data[cols].to_csv(new_name + ".csv", index=False)
            elif format_l in ("dta", "stata"):
                # use version=119 to avoid the following error
                # >> UnicodeEncodeError: 'latin-1' codec can't encode
                self.__data[cols].to_stata(new_name + ".dta", write_index=False, version=119)
            elif format_l in ("excel", "xlsx"):
                self.__data[cols].to_excel(new_name + ".xlsx", index=False)
            else:
                print(f"\nSorry, the format {format} is not supported.")
                return
            
        out_col_list = [columns]
        if want_sep:
            out_col_list.append([
                elm for elm in columns if not (elm.startswith('seg_lon_') or elm.startswith('seg_lat_'))
            ])

        for c in out_col_list:
            filename = new_name
            if len(c) == max_col_cnt:
                filename = new_name + "_w_exact_pt"
            output(c, filename)
            

        print("FINISHED.")
        return

def main():
    filepath = ".\\Rental_geocode\\不動產租賃_master.dta"

    start1 = time.time()

    gp = GeoProcessor(pd.read_stata(filepath))
    gp._make_shorter()
    gp.coord_twd97towgs84()

    for i in range(4):
        gp.find_nearest_point(ref_file_ver=i+1)
    
    start2 = time.time()

    # gp.show_data(is_thin=True)
    # gp.output_data("dta", want_sep=True)

    end = time.time()

    print(f"\nIt took {datetime.timedelta(seconds=round(start2-start1))} (H:MM:SS) to process data.")
    print(f"It took {datetime.timedelta(seconds=round(end-start2))} (H:MM:SS) to output data.")


if __name__ == "__main__":
    main()