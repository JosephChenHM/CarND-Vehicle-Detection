import cv2
import numpy as np

class Line:
    def __init__(self):
        # if the first frame of video has been processed
        self.first_frame_processed = False  
        
        self.img = None
        
        self.avg_left_fit_coeffs = None
        self.avg_right_fit_coeffs = None
        self.mse_tolerance = 0.001
        self.left_fit = [np.array([False])] 
        self.right_fit = [np.array([False])] 
        self.his_left_fit = []
        self.his_right_fit = []
        self.y_eval = 700
        self.midx = 640
        self.ym_per_pix = 3.0/100.0 # meters per pixel in y dimension
        self.xm_per_pix = 3.7/600.0 # meters per pixel in x dimension
        self.curvature = 0
       
    def update_avg_coeffe(self):
        """
        Average curve fitting coefficient
        """
        left_coeffs = self.his_left_fit[len(self.his_left_fit)-10:]
        #print(left_coeffs)
        right_coeffs = self.his_right_fit[len(self.his_right_fit)-10:]
        #print(right_coeffs)
        if len(left_coeffs)>0:
            avg=0
            for coeff in left_coeffs:
                avg +=np.array(coeff)
            avg = avg / len(left_coeffs)
            self.avg_left_fit_coeffs = avg
            #print(self.avg_left_fit_coeffs)
        if len(right_coeffs)>0:
            avg=0
            for coeff in right_coeffs:
                avg +=np.array(coeff)
            avg = avg / len(right_coeffs)
            self.avg_right_fit_coeffs = avg
            #print(self.avg_right_fit_coeffs)
            
    def update_fits(self, left_fit, right_fit):
        """
        Update the co-efficients of fitting polynomial
        """
        if self.first_frame_processed:
            left_error = ((self.left_fit[0] - left_fit[0])).mean(axis=None)
            right_error = ((self.right_fit[0] - right_fit[0])).mean(axis=None)
            #print(left_error)
            #print(right_error)
            if abs(left_error) < self.mse_tolerance:
                #self.left_fit = (0.75 * self.left_fit + 0.25 * left_fit)
                self.left_fit = left_fit
                self.his_left_fit.append(self.left_fit)
                #print("L bad fit")
            if abs(right_error) < self.mse_tolerance:
                #self.right_fit = (0.75 * self.right_fit + 0.25 * right_fit)
                self.right_fit = right_fit
                self.his_right_fit.append(self.right_fit)
                #print("R bad fit")
        else:
            self.right_fit = right_fit
            self.left_fit = left_fit
            self.his_left_fit.append(left_fit)
            self.his_right_fit.append(right_fit)
        
        self.update_avg_coeffe()
        #print(self.avg_right_fit_coeffs)
        self.update_curvature(self.avg_right_fit_coeffs)
     
     
    def update_curvature(self, fit):
        """
        Update radius of curvature
        """
        y1 = (2*fit[0]*self.y_eval + fit[1])*self.xm_per_pix/self.ym_per_pix
        y2 = 2*fit[0]*self.xm_per_pix/(self.ym_per_pix**2)
        curvature = ((1 + y1*y1)**(1.5))/np.absolute(y2)
        
        if self.first_frame_processed:
            self.curvature = curvature
        
        elif np.absolute(self.curvature - curvature) < 500:
            self.curvature = 0.75*self.curvature + 0.25*(((1 + y1*y1)**(1.5))/np.absolute(y2)) 

    def get_position_from_center(self):
        """
        Get distance from center
        """
        x_left_pix = self.avg_left_fit_coeffs[0]*(self.y_eval**2) + self.avg_left_fit_coeffs[1]*self.y_eval + self.avg_left_fit_coeffs[2]
        x_right_pix = self.avg_right_fit_coeffs[0]*(self.y_eval**2) + self.avg_right_fit_coeffs[1]*self.y_eval + self.avg_right_fit_coeffs[2]
        
        return (self.midx - (x_left_pix + x_right_pix)/2.0) * self.xm_per_pix