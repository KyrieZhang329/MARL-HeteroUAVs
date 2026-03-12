import numpy as np

class GridMapScan:
    def __init__(self,map_size=20.0,grid_num=20):
        self.map_size = map_size
        self.grid_num = grid_num
        self.cell_size = map_size/grid_num
        self.grid = np.zeros((grid_num,grid_num),dtype=np.int8)

    def reset(self):
        self.grid.fill(0)

    def trans_coordinates(self,pos):
        offset = self.map_size/2
        x = np.clip(pos[0],-offset,offset-0.01)
        y = np.clip(pos[1],-offset,offset-0.01)
        col=int((x+offset)/self.cell_size)
        row=int((y-offset)/self.cell_size)
        return col,row

    def update_coverage(self,drone_pos,scan_r):
        r_in_cells = int(scan_r/self.cell_size)
        center_col,center_row = self.trans_coordinates(drone_pos)
        newly_covered_count = 0

        for r in range(center_row-r_in_cells,center_row+r_in_cells+1):
            for c in range(center_col-r_in_cells,center_col+r_in_cells+1):
                if 0<=r<self.grid_num and 0<=c<self.grid_num:
                    grid_center_x = c*self.cell_size-self.map_size/2+self.cell_size/2
                    grid_center_y = r*self.cell_size-self.map_size/2+self.cell_size/2
                    drone_pos_2d = drone_pos[:2]
                    distance = np.linalg.norm(drone_pos_2d-np.array([grid_center_x,grid_center_y]))

                    if distance < scan_r:
                        if self.grid[r,c]==0:
                            self.grid[r,c] = 1
                            newly_covered_count += 1
        return newly_covered_count

    def get_coverage_rate(self):
        return np.sum(self.grid)/(self.grid_num**2)