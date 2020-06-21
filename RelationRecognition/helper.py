import math

import numpy


class calculate_helper:
    def __init__(self):
        self.compass_brackets = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]

    def cal_sub2obj_degree(self, sub_x, obj_x, sub_y, obj_y):
        delta_x = obj_x - sub_x
        delta_y = obj_y - sub_y
        degrees_temp = math.atan2(delta_x,delta_y)/math.pi*180

        if degrees_temp<0:
            degrees_final = 360 + degrees_temp
        else:
            degrees_final = degrees_temp

        return degrees_final

    def cal_sub2obj_direction(self, sub_x, obj_x, sub_y, obj_y):
        degree = self.cal_sub2obj_degree(sub_x, obj_x, sub_y, obj_y)

        idx = round(degree / (360. / len(self.compass_brackets)))
        compass = self.compass_brackets[idx % len(self.compass_brackets)]
        return self.compass_brackets.index(compass) # one_hot_coding 말고 그냥 바로 사용할 수 있음

    def dir_convert2_one_hot(self, ind):
        compass_one_hot = numpy.zeros(len(self.compass_brackets))
        compass_one_hot[ind] = 1
        return compass_one_hot

    def cal_bbox_center(self, xmin, ymin, xmax, ymax):
        coords = (xmin, xmax, ymin, ymax)
        center_x, center_y = (numpy.average(coords[:2]), numpy.average(coords[2:]))
        return (center_x, center_y)

    def cal_bbox_WnH(self, subject_loc, object_loc, add_area_ratio):

        self.indexSearch = ["XMIN", "YMIN", "XMAX", "YMAX"]
        WnH_ratio_builder = numpy.zeros(len(self.indexSearch)*2)
        sub_bbox = (subject_loc[0], subject_loc[1], subject_loc[2], subject_loc[3]) #[XMIN, YMIN, XMAX, YMAX] [555, 279, 887, 680]
        obj_bbox = (object_loc[0], object_loc[1], object_loc[2], object_loc[3])     #[XMIN, YMIN, XMAX, YMAX] [6, 160, 424, 746]

        sub_x_len = sub_bbox[self.indexSearch.index("XMAX")] - sub_bbox[self.indexSearch.index("XMIN")]
        sub_y_len = sub_bbox[self.indexSearch.index("YMAX")] - sub_bbox[self.indexSearch.index("YMIN")]

        obj_x_len = obj_bbox[self.indexSearch.index("XMAX")] - obj_bbox[self.indexSearch.index("XMIN")]
        obj_y_len = obj_bbox[self.indexSearch.index("YMAX")] - obj_bbox[self.indexSearch.index("YMIN")]

        if sub_x_len > sub_y_len:
            if sub_x_len > sub_y_len*2:
                WnH_ratio_builder[0] = 1
            else:
                WnH_ratio_builder[1] = 1
        else:
            if sub_y_len > sub_x_len*2:
                WnH_ratio_builder[3] = 1
            else:
                WnH_ratio_builder[2] = 1

        if obj_x_len > obj_y_len:
            if obj_x_len > obj_y_len*2:
                WnH_ratio_builder[4] = 1
            else:
                WnH_ratio_builder[5] = 1
        else:
            if obj_y_len > obj_x_len*2:
                WnH_ratio_builder[7] = 1
            else:
                WnH_ratio_builder[6] = 1

        if add_area_ratio:
            sub2obj_area = (sub_x_len*sub_y_len)/(obj_x_len*obj_y_len)
            sub2obj_area = round(sub2obj_area, 2)
            WnH_ratio_builder = numpy.append(WnH_ratio_builder, numpy.array(sub2obj_area))

        return WnH_ratio_builder

# if __name__ == '__main__':
#     helper = calculate_helper()
    # print(helper.cal_sub2obj_degree(4.1,2.1,-1,-1.1))
    #print(helper.cal_sub2obj_direction(4.1,2.1,-1,-1.1))
    # print(helper.cal_bbox_center(230,72,415,661))
    # direction = helper.cal_sub2obj_direction(4.1,2.1,-1,-1.1)
    # result = helper.convert2_one_hot(direction)
    # print(result)
    #
    # idx = round(288.356 / (360. / len(helper.compass_brackets)))
    # print(helper.compass_brackets[idx % len(helper.compass_brackets)])

    # subject_bbox = [555, 279, 887, 680]
    # object_bbox = [6, 160, 424, 746]
    # result1 = helper.cal_bbox_WnH(subject_bbox, object_bbox, 0)
    # result2 = helper.cal_bbox_WnH(subject_bbox, object_bbox, 1)
    # print(helper.cal_bbox_WnH(subject_bbox, object_bbox, 0))
    # print(helper.cal_bbox_WnH(subject_bbox, object_bbox, 1))
