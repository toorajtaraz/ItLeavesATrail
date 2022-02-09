import cv2
import numpy as np
from time import gmtime, strftime
from math import pi, atan2

#NEEDED DATA
TWO_PI = 2 * pi

#Remember to change this to the correct path
data_folder_path = '$HOME/Documents/ItLeavesATrail/data/'

books = [
    'Angels and Demons', 
    # 'Anne of Green Gables & Anne of Avonlea', 
    'David Copperfield', 
    # 'Dracula', 
    # 'Pickwick Papers', 
    # 'To Kill a Mockingbird',
    # 'Tom Sawyer and Huckleberry Finn',
    # 'Twenty Thousand Leagues Under the Sea',
    # 'Twilight - New Moon',
    # 'Twilight - Eclipse'
]

format_extentions = [
    '.jpg',
    '.mp4'
]

base_clips = [
    '0.MOV',
    '1.MOV'
]

#LOADING BINARIES
book_covers = []

for book in books:
    path = data_folder_path + book + format_extentions[0]
    book_covers.append(cv2.imread(path))

trailers = []

for trailer in books:
    path = data_folder_path + trailer + format_extentions[1]
    trailers.append(path)


# for cover in book_covers:
#     cv2.imshow("img", cover)
#     cv2.waitKey(0)

#LOAD DETECTORS
sift = cv2.SIFT_create()
surf = cv2.xfeatures2d.SURF_create()
orb = cv2.ORB_create(nfeatures=1500)

#FEATURE DETECTION AND EXTRACTION AND MATCHING
def detect(img, detector="sift"):
    if detector == "sift":
        return sift.detectAndCompute(img, None)
    elif detector == "surf":
        return surf.detectAndCompute(img, None)
    elif detector == "orb":
        return orb.detectAndCompute(img, None)
    return None

def show_points(img, keypoints):
    cv2.imshow("POINTS", cv2.drawKeypoints(img, keypoints, None))

def matcher(desc1, desc2, matches_thresh=0.10, match_algo="brute", detector="sift"):
    if match_algo == "brute":
        if detector == "sift" or detector == "surf":
            bfm = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        elif detector == "orb":
            bfm = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bfm.match(desc1, desc2)
        matches = sorted(matches, key = lambda x : x.distance)
        return matches[:round(len(matches) * matches_thresh)]
    elif match_algo == "flann":
        FLANN_INDEX_KDTREE = 0
        FLANN_INDEX_LSH = 6
        if detector == "sift" or detector == "surf":
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        elif detector == "orb":
            index_params= dict(algorithm = FLANN_INDEX_LSH,
                            table_number = 6,
                            key_size = 12,
                            multi_probe_level = 1)
        search_params = dict(checks=100)
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(desc1, desc2, k=2)
        good_matches = []
        try:
            for m,n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        except Exception:
            return None              
        return good_matches
    else:
        return None

def show_matches(img1, kp1, img2, kp2, matches, flags, vars):
    cv2.imshow("MATCHES", cv2.drawMatches(img1, kp1, img2, kp2, matches, None, **vars))

def is_convex_polygon(polygon):
    #this function was retrived from https://github.com/jamiebull1/geomeppy
    """Return True if the polynomial defined by the sequence of 2D
    points is 'strictly convex': points are valid, side lengths non-
    zero, interior angles are strictly between zero and a straight
    angle, and the polygon does not intersect itself.

    NOTES:  1.  Algorithm: the signed changes of the direction angles
                from one side to the next side must be all positive or
                all negative, and their sum must equal plus-or-minus
                one full turn (2 pi radians). Also check for too few,
                invalid, or repeated points.
            2.  No check is explicitly done for zero internal angles
                (180 degree direction-change angle) as this is covered
                in other ways, including the `n < 3` check.
    """
    try:  # needed for any bad points or direction changes
        # Check for too few points
        if len(polygon) < 3:
            return False
        # Get starting information
        old_x, old_y = polygon[-2]
        new_x, new_y = polygon[-1]
        new_direction = atan2(new_y - old_y, new_x - old_x)
        angle_sum = 0.0
        # Check each point (the side ending there, its angle) and accum. angles
        for ndx, newpoint in enumerate(polygon):
            # Update point coordinates and side directions, check side length
            old_x, old_y, old_direction = new_x, new_y, new_direction
            new_x, new_y = newpoint
            new_direction = atan2(new_y - old_y, new_x - old_x)
            if old_x == new_x and old_y == new_y:
                return False  # repeated consecutive points
            # Calculate & check the normalized direction-change angle
            angle = new_direction - old_direction
            if angle <= -pi:
                angle += TWO_PI  # make it in half-open interval (-Pi, Pi]
            elif angle > pi:
                angle -= TWO_PI
            if ndx == 0:  # if first time through loop, initialize orientation
                if angle == 0.0:
                    return False
                orientation = 1.0 if angle > 0.0 else -1.0
            else:  # if other time through loop, check orientation is stable
                if orientation * angle <= 0.0:  # not both pos. or both neg.
                    return False
            # Accumulate the direction-change angle
            angle_sum += angle
        # Check that the total number of full turns is plus-or-minus 1
        return abs(round(angle_sum / TWO_PI)) == 1
    except (ArithmeticError, TypeError, ValueError):
        return False  # any exception means not a proper convex polygon

def overlay_on_frame(cover, frame, matches, cover_det, frame_det, MIN_MATCH_COUNT = 35, debug=False):
    if len(matches) > MIN_MATCH_COUNT:
        src_pts = []
        dst_pts = []
        kp_cover = cover_det[0]
        kp_frame = frame_det[0]
        for match in matches:
            src_pts.append(kp_cover[match.queryIdx].pt)
            dst_pts.append(kp_frame[match.trainIdx].pt)

        src_pts = np.float32(src_pts)
        dst_pts = np.float32(dst_pts)

        trans_matrix, mask = cv2.findHomography(src_pts, 
                dst_pts, 
                cv2.RANSAC,
                5.0)
        matchesMask = mask.ravel().tolist()

        h,w, _ = cover.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

        try:
            dst = cv2.perspectiveTransform(pts, trans_matrix).reshape(-1, 2).tolist()
            if not is_convex_polygon(dst):
                return None
        except Exception as e:
            if debug:
                print(e)
            return None
        result = cv2.warpPerspective(cover, 
                trans_matrix, 
                (frame.shape[1],frame.shape[0]), 
                dst=frame, 
                borderMode=cv2.BORDER_TRANSPARENT
            )
        return result, matchesMask
    else:
        return None

#LOGIC
def detection_pipeline(debug=False, MIN_MATCH_COUNT = 35, detector="orb", match_algo="flann"):
    for _index, wanted_book in enumerate(book_covers):
        cover_det = detect(wanted_book, detector=detector)
        trailer = trailers[_index]
        for base_clip in base_clips:
            cap = cv2.VideoCapture(data_folder_path + base_clip)

            trailer_cap = cv2.VideoCapture(trailer)
            base_clip_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            trailer_frames = int(trailer_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            which_has_more_frames = 1 if base_clip_frames > trailer_frames else 2
            if which_has_more_frames == 1:
                rate_frame = np.ceil(base_clip_frames / trailer_frames)
            else:
                rate_frame = np.ceil(trailer_frames / base_clip_frames)
            output_video = cv2.VideoWriter(
                                "{}_{}_{}_{}_trailer_on_top_{}.mp4".format(detector, match_algo, base_clip, books[_index], strftime("%Y-%m-%d_%H:%M:%S", gmtime())), 
                                fourcc = cv2.VideoWriter_fourcc(*'mp4v'),
                                fps = 24,
                                frameSize = (
                                    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                )
                            )
            c = 0
            counter = 0
            prev = wanted_book
            while(cap.isOpened()):
                try:
                    ret, frame = cap.read()             
                    frame_det = detect(frame, detector=detector)
                    cover_desc = cover_det[1]
                    frame_desc = frame_det[1]
                    res = matcher(cover_desc, frame_desc, match_algo=match_algo, detector=detector)
                    try:
                        if which_has_more_frames == 1:
                            if counter == rate_frame or counter == 0:
                                counter = 0 if counter == rate_frame else 1
                                ret, trailer_frame = trailer_cap.read()
                                trailer_frame = cv2.resize(trailer_frame, 
                                                    (np.ceil(trailer_frame.shape[1] / wanted_book.shape[1]), 
                                                    np.ceil(trailer_frame.shape[0] / wanted_book.shape[0]))
                                            )
                            try:
                                result, matchesMask = overlay_on_frame(trailer_frame, frame, res, cover_det, frame_det, MIN_MATCH_COUNT)
                                output_video.write(result)
                                prev = (res, frame_det)
                            except:
                                result, matchesMask = overlay_on_frame(trailer_frame, frame, prev[0], cover_det, prev[1], MIN_MATCH_COUNT)
                                output_video.write(result)
                                if debug:
                                    print("missed_frame", c)
                                    c += 1
                            if debug:
                                cv2.imshow("res", result)
                                cv2.waitKey(0)
                            counter += 1 if counter >= 1 else 0
                        else:
                            for _ in range(int(rate_frame)):
                                ret, trailer_frame = trailer_cap.read()
                                needed_size = (
                                    int((wanted_book.shape[1] / trailer_frame.shape[1]) * trailer_frame.shape[1]),
                                    int((wanted_book.shape[0] / trailer_frame.shape[0]) * trailer_frame.shape[0])
                                )
                                trailer_frame = cv2.resize(trailer_frame, 
                                                           needed_size,
                                                           interpolation = cv2.INTER_CUBIC
                                                )
                                try:
                                    result, matchesMask = overlay_on_frame(trailer_frame, frame, res, cover_det, frame_det, MIN_MATCH_COUNT)
                                    output_video.write(result)
                                    prev = (res, frame_det)
                                except:
                                    result, matchesMask = overlay_on_frame(trailer_frame, frame, prev[0], cover_det, prev[1], MIN_MATCH_COUNT)
                                    output_video.write(result)
                                    if debug:
                                        print("missed_frame", c)
                                        c += 1
                                if debug:
                                    cv2.imshow("res", result)
                                    cv2.waitKey(0)
                    except Exception as e:
                        if debug:
                            print(e)
                        continue
                    if debug:
                        draw_params = dict(singlePointColor = None,
                                        matchesMask = matchesMask,
                                        flags = 2
                                    )
                        show_matches(wanted_book, cover_det[0], frame, frame_det[0], res, 2, draw_params)
                        cv2.waitKey(0)
                except Exception as e:
                    if debug:
                        print(e)
                    break
            output_video.release()
            trailer_cap.release()
            cap.release()
    cv2.destroyAllWindows()

detection_pipeline(detector="orb", MIN_MATCH_COUNT=25)
detection_pipeline(detector="sift", MIN_MATCH_COUNT=20)
detection_pipeline(detector="surf", MIN_MATCH_COUNT=5)
#ENDING