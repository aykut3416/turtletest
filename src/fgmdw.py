#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped,Twist
from nav_msgs.msg import Odometry,OccupancyGrid
from tf.transformations import euler_from_quaternion
import math as m
import numpy as np
import random
from sklearn.neighbors import KDTree
import warnings
warnings.filterwarnings("ignore")

#initial values
initial_pos=[0,0,0]
#capraz icin
goal_pos=[9.0,7.0]
#duz icin
# goal_pos=[5.,0.]
scan_data=[]
min_d=2.5
alfa=2.0
max_range=2.0
k_vel=0.7
k_ori=0.3
best_v=0
best_w=0
kdtree,setups,newarr=[],[],[]


#subscribe the occupancy grid map
def get_map(msg):
    global d2, setups, newarr, best_v, best_w, kdtree
    d2 = zip(*[iter(msg.data)] * 60)
    setups = [msg.info.origin.position.x, msg.info.origin.position.y, msg.info.resolution]
    #calculate obstacle coordinates
    newarr = [[setups[0] + i * setups[2] + 0.025, setups[1] + j * setups[2] + 0.025] for j in range(60) for i in range(60) if 100 > d2[j][i] > 50]
    if not newarr==[]:
        #use fgm to calculate maximum gap center
        tree,midp = find_max_gap(newarr)
        dist, ind = tree.query([initial_pos[0], initial_pos[1]], k=1)
        d_min = dist[0][0]

        gap_angle = m.atan2(midp[1] - initial_pos[1], midp[0] - initial_pos[0])
        g_angle = m.atan2(m.sin(gap_angle - initial_pos[2]), m.cos(gap_angle - initial_pos[2]))
        goal_angle = m.atan2(goal_pos[1] - initial_pos[1], goal_pos[0] - initial_pos[0])
        h_angle = m.atan2(m.sin(goal_angle - initial_pos[2]), m.cos(goal_angle - initial_pos[2]))
        #fgm guide angle calculation
        fgm_angle = ((alfa / d_min) * g_angle + h_angle) / ((alfa / d_min) + 1)
        last_angle = m.atan2(m.sin(initial_pos[2] + fgm_angle), m.cos(initial_pos[2] + fgm_angle))
    else:
        d_min=2.5
        last_angle = m.atan2(goal_pos[1] - initial_pos[1], goal_pos[0] - initial_pos[0])
    # go to dwa function with guide angle
    best_v, best_w = dwa_func(last_angle)
    


def find_max_gap(pts):
    kdtree = KDTree(pts, leaf_size=50)
    angles=[m.atan2(j-initial_pos[1],i-initial_pos[0]) for i,j in pts]
    angles=[m.atan2(m.sin(i-initial_pos[2]),m.cos(i-initial_pos[2])) for i in angles]
    #polar histogram
    polarlist=[[j,pts[i]] for i,j in enumerate(angles) if (m.pi/2.)>=j>=(-m.pi/2.)]
    polarlist=sorted(polarlist)
    # calculate gap sizes and center points
    if not polarlist == []:
        meas = [[m.hypot(polarlist[i + 1][1][0] - polarlist[i][1][0], polarlist[i + 1][1][1] - polarlist[i][1][1]),
                 [(polarlist[i + 1][1][0] + polarlist[i][1][0]) / 2.,
                  (polarlist[i + 1][1][1] + polarlist[i][1][1]) / 2.]]
                for i in range(1, len(polarlist[1:])) if abs(polarlist[i + 1][0] - polarlist[i][0]) > 0.1]
        if polarlist[0][0] + m.pi / 2. > 0.1:
            dist = m.hypot(polarlist[0][1][0] - initial_pos[0], polarlist[0][1][1] - initial_pos[1])
            p1 = [initial_pos[0] + m.cos(-m.pi / 2. + initial_pos[2]) * dist,
                  initial_pos[1] + m.sin(-m.pi / 2. + initial_pos[2]) * dist]
            mid_p = [(p1[0] + polarlist[0][1][0]) / 2., (p1[1] + polarlist[0][1][1]) / 2.]
            g_size = m.hypot(p1[0] - polarlist[0][1][0], p1[1] - polarlist[0][1][1])
            meas.append([g_size, mid_p])
        if m.pi / 2. - polarlist[-1][0] > 0.1:
            dist = m.hypot(polarlist[-1][1][0] - initial_pos[0], polarlist[-1][1][1] - initial_pos[1])
            p1 = [initial_pos[0] + m.cos(m.pi / 2. + initial_pos[2]) * dist,
                  initial_pos[1] + m.sin(m.pi / 2. + initial_pos[2]) * dist]
            mid_p = [(p1[0] + polarlist[-1][1][0]) / 2., (p1[1] + polarlist[-1][1][1]) / 2.]
            g_size = m.hypot(p1[0] - polarlist[-1][1][0], p1[1] - polarlist[-1][1][1])
            meas.append([g_size, mid_p])
        meas = sorted(meas)
        return kdtree, meas[-1][1]
    else:
        return kdtree, goal_pos

#-------------------reachable velocities----------------------------
def dwa_func(t_angle):
    scores=[]
    v_reachable = np.linspace(0., 0.4, 9)
    w_reachable = np.linspace(-1., 1., 11)
    for i in v_reachable:
        for j in w_reachable:
            check,ori=motion_model(initial_pos,i,j)
            # if velocity is admissible
            if check:
                ori_diff=abs(m.atan2(m.sin(ori-t_angle),m.cos(ori-t_angle)))
                # objective function calculation
                if m.hypot(goal_pos[0]-initial_pos[0],goal_pos[1]-initial_pos[1])>0.5 or i<0.05:
                    scores.append([k_vel*(i/v_reachable[-1])+k_ori*(1-(ori_diff/m.pi)),i,j])
                elif i>0.05:
                    scores.append([k_vel*(1-i/v_reachable[-1])+k_ori*(1-(ori_diff/m.pi)),i,j])
    scores=sorted(scores)
    #return with best anglular, linear velocity pair
    return scores[-1][1],scores[-1][2]

# calculation of differential drive kinematics
def motion_model(initial,V,w):
    ori = np.linspace(0, w, 5) + initial[2]
    dx = np.cos(np.linspace(0, w, 10)[range(1, 10, 2)] + initial[2]) * 0.2 * V
    dy = np.sin(np.linspace(0, w, 10)[range(1, 10, 2)] + initial[2]) * 0.2 * V
    x = [initial[0] + np.sum(dx[:i + 1]) for i in range(5)]
    y = [initial[1] + np.sum(dy[:i + 1]) for i in range(5)]
    pts = [[x[i],y[i]] for i in range(5)]
    if not kdtree==[]:
        if check_path(pts) and stop_check(pts,V,w):
            return True,m.atan2(m.sin(ori[-1]),m.cos(ori[-1]))
        else:
            return False,0
    else:
        return True,m.atan2(m.sin(ori[-1]),m.cos(ori[-1]))

#collision check
def check_path(pt):
    random.shuffle(pt)
    for i,j in pt:
        pix=int((i-setups[0])/setups[2])
        piy=int((j-setups[1])/setups[2])
        if d2[piy][pix]>50:
            return False
    return True

# return true if pair is able to stop collision
def stop_check(pts_sim,V_sim,w_sim):
    d=[]
    for l,n in pts_sim:
        dist, ind = kdtree.query([l,n], k=1)
        d.append(dist[0][0])
    if V_sim < m.sqrt(2. * min(d) * .4) and abs(w_sim) < m.sqrt(2 * min(d) * 1.66):
        return True
    else:
        return False
#--------------------------------------------------------
    
def get_odom(msg2):
    global initial_pos
    quat=msg2.pose.pose.orientation
    roll,pitch,yaw=euler_from_quaternion([quat.x,quat.y,quat.z,quat.w])
    initial_pos=[round(msg2.pose.pose.position.x,3),round(msg2.pose.pose.position.y,3),round(yaw,3)]

def get_goal(msg3):
    global goal_pos
    goal_pos=[round(msg3.pose.position.x,3),round(msg3.pose.position.y,3)]
    
    
if __name__=='__main__':
    rospy.init_node('fgmdw_node', anonymous=True)
    pub = rospy.Publisher('/cmd_vel_mux/input/safety_controller', Twist, queue_size=1)
    rospy.Subscriber('/move_base/local_costmap/costmap', OccupancyGrid, get_map)
    rospy.Subscriber("/odom", Odometry, get_odom)
    rospy.Subscriber("/move_base_simple/goal", PoseStamped, get_goal)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        if m.hypot(initial_pos[0]-goal_pos[0],initial_pos[1]-goal_pos[1])>0.15 and not goal_pos==[0,0]:
            a=Twist()
            a.linear.x=best_v
            a.angular.z=best_w
            pub.publish(a)
        else:
            pub.publish(Twist())
        rate.sleep()
