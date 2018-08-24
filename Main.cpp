#include<opencv2\opencv.hpp> 
#include <stdio.h>  
#include <cstring>
#include <string>
#include <math.h>  
#include <iostream>
#include <iomanip>
#include <math.h>
#include <fstream>
#include <thread>  
#include <mutex>
#include <ctime>
#include <memory>
#include <random>
#include <map>  
#include <unordered_map>  

/* Render by Tongda Xu for Computer Graphic Class in Tsinghua University, 2018 Spring, Taught by S M Hu and S Yang */
/* Core functionality:*/
/*
Objects:
Basic Geometry (Plane, Sphere)
Load Mesh from *.obj file
Phong Material

Scene:
Standard Camera
Scene Management

Render:
Ray Casting
Ray Tracing
Path Tracing
Photon Mapping

Accelerction:
AABB
MultiThread (Path Tracing, Photon Mapping)
BspTree (Path Tracing Mesh, Photon Mapping Photon)
*/

/* define namespace */
using namespace cv;

/* define Global Variable */
#define THREAD 6
#define MAX_DEPTH 8
#define M_PI 3.141592653589
#define INFINITY 1e8
#define BIAS 1e-4
#define PTSAMPLINGLEVEL 40
#define BSPDEPTH 5
#define ENGINE "PHOTONMAPPING"
#define IFBSPTREE true
#define PHOTONNUM 20000000
#define GATHERRADIUS 0.5
#define LIGHTFACTOR 0.0025
#define BATCHSIZE 2000000
#define BATCHNUM 100

float GxMin = INFINITY;
float GxMax = -INFINITY;
float GyMin = INFINITY;
float GyMax = -INFINITY;
float GzMin = INFINITY;
float GzMax = -INFINITY;

/* utility class start */

class Vector3 {

public:

	float x;
	float y;
	float z;

	Vector3() {
		x, y, z = 0;
	}

	Vector3(float a) {
		x = a;
		y = a;
		z = a;
	}

	Vector3(float a, float b, float c) {
		x = a;
		y = b;
		z = c;
	}

	Vector3(const Vector3 &v) {
		x = v.x;
		y = v.y;
		z = v.z;
	}

	~Vector3() {}

	Vector3 operator * (const float a) const {
		return Vector3(x*a, y*a, z*a);
	}

	Vector3 operator - () const {
		return Vector3(-x, -y, -z);
	}

	Vector3 operator + (const Vector3 &v) const {
		return Vector3(x + v.x, y + v.y, z + v.z);
	}

	Vector3 operator - (const Vector3 &v) const {
		return Vector3(x - v.x, y - v.y, z - v.z);
	}

	Vector3 operator * (const Vector3 &v) const {
		return Vector3(y*v.z - z * v.y, z*v.x - x * v.z, x*v.y - y * v.x);
	}

	float dot(const Vector3 &v) const {
		return  x * v.x + y * v.y + z * v.z;
	}

	long double length() const {
		return sqrtl(x*x + y * y + z * z);
	}

	Vector3 normal() {
		return *this*(1 / length());
	}

	void toString() {
		std::cout << "[" << x << ", " << y << ", " << z << "]" << std::endl;
	}

};

class BoundingBox {

public:
	float minX;
	float maxX;
	float minY;
	float maxY;
	float minZ;
	float maxZ;
	float tmin[3];
	float tmax[3];

	Vector3 p1;
	Vector3 p2;
	Vector3 p3;
	Vector3 p4;
	Vector3 p5;
	Vector3 p6;
	Vector3 p7;
	Vector3 p8;

	BoundingBox() {}
	BoundingBox(float sX, float bX, float sY, float bY, float sZ, float bZ) {

		minX = sX;
		maxX = bX;
		minY = sY;
		maxY = bY;
		minZ = sZ;
		maxZ = bZ;

		p1 = Vector3(minX, minY, minZ);
		p2 = Vector3(minX, maxY, minZ);
		p3 = Vector3(maxX, maxY, minZ);
		p4 = Vector3(maxX, minY, minZ);

		p5 = Vector3(minX, minY, maxZ);
		p6 = Vector3(minX, maxY, maxZ);
		p7 = Vector3(maxX, maxY, maxZ);
		p8 = Vector3(maxX, minY, maxZ);

	}
	~BoundingBox() {}

	bool ifPointInside(const Vector3 &pointToTest) {

		if (pointToTest.x > maxX || pointToTest.x < minX) {
			return false;
		}

		if (pointToTest.y > maxY || pointToTest.y < minY) {
			return false;
		}

		if (pointToTest.z > maxZ || pointToTest.z < minZ) {
			return false;
		}

		return true;
	}

	float PointDist(Vector3 pointToTest) {

		//pointToTest.toString();

		float distX1 = minX - pointToTest.x;
		float distX2 = maxX - pointToTest.x;

		float distY1 = minY - pointToTest.y;
		float distY2 = maxY - pointToTest.y;

		float distZ1 = minZ - pointToTest.z;
		float distZ2 = maxZ - pointToTest.z;

		float distX = std::min(abs(distX1), abs(distX2));
		float distY = std::min(abs(distY1), abs(distY2));
		float distZ = std::min(abs(distZ1), abs(distZ2));
		
		if (distX1*distX2 < 0) {
			distX = 0;
		}

		if (distY1*distY2 < 0) {
			distY = 0;
		}

		if (distZ1*distZ2 < 0) {
			distZ = 0;
		}

		float finalDist = sqrt(distX*distX + distY * distY + distZ * distZ);

		return finalDist;

	}

	bool bxIntersect(const Vector3 &BeginPoint, Vector3 Direction) {

		if (ifPointInside(BeginPoint)) {
			return true;
		}

		Direction = Direction.normal();

		if (Direction.x == 0) {
			tmin[0], tmax[0] = INFINITY;
		}
		else {
			float x1 = ((minX - BeginPoint.x) / Direction.x);
			float x2 = ((maxX - BeginPoint.x) / Direction.x);

			tmin[0] = x1 <= x2 ? x1 : x2;
			tmax[0] = x1 <= x2 ? x2 : x1;
		}

		if (Direction.y == 0) {
			tmin[1], tmax[1] = INFINITY;
		}
		else {
			float y1 = ((minY - BeginPoint.y) / Direction.y);
			float y2 = ((maxY - BeginPoint.y) / Direction.y);

			tmin[1] = y1 <= y2 ? y1 : y2;
			tmax[1] = y1 <= y2 ? y2 : y1;
		}

		if (Direction.z == 0) {
			tmin[2], tmax[2] = INFINITY;
		}
		else {
			float z1 = ((minZ - BeginPoint.z) / Direction.z);
			float z2 = ((maxZ - BeginPoint.z) / Direction.z);

			tmin[2] = z1 <= z2 ? z1 : z2;
			tmax[2] = z1 <= z2 ? z2 : z1;
		}

		float gTmin = std::max((std::max(tmin[0], tmin[1])), tmin[2]);
		float gTmax = std::min(std::min(tmax[0], tmax[1]), tmax[2]);

		if (std::max((std::max(tmin[0], tmin[1])), tmin[2])<std::min(std::min(tmax[0], tmax[1]), tmax[2])) {
			return true;

		}
		else {
			return false;
		}
	}

	float bspIntersect(const Vector3 &BeginPoint, const Vector3 &Direction) {

		if (ifPointInside(BeginPoint)) {
			// if inside, we do not need to calculate the exact t, just return 0 since it is the smallest possible result
			return 0;
		}

		if (Direction.x == 0) {
			tmin[0], tmax[0] = INFINITY;
		}
		else {
			float x1 = ((minX - BeginPoint.x) / Direction.x);
			float x2 = ((maxX - BeginPoint.x) / Direction.x);

			tmin[0] = x1 <= x2 ? x1 : x2;
			tmax[0] = x1 <= x2 ? x2 : x1;
		}

		if (Direction.y == 0) {
			tmin[1], tmax[1] = INFINITY;
		}
		else {
			float y1 = ((minY - BeginPoint.y) / Direction.y);
			float y2 = ((maxY - BeginPoint.y) / Direction.y);

			tmin[1] = y1 <= y2 ? y1 : y2;
			tmax[1] = y1 <= y2 ? y2 : y1;
		}

		if (Direction.z == 0) {
			tmin[2], tmax[2] = INFINITY;
		}
		else {
			float z1 = ((minZ - BeginPoint.z) / Direction.z);
			float z2 = ((maxZ - BeginPoint.z) / Direction.z);

			tmin[2] = z1 <= z2 ? z1 : z2;
			tmax[2] = z1 <= z2 ? z2 : z1;
		}

		float Gtmin = std::max(std::max(tmin[0], tmin[1]), tmin[2]);
		float Gtmax = std::min(std::min(tmax[0], tmax[1]), tmax[2]);
		if (std::max((std::max(tmin[0], tmin[1])), tmin[2])<std::min(std::min(tmax[0], tmax[1]), tmax[2])) {
			return Gtmin;
		}
		else {
			// if no intersect, return -1
			return -1;
		}
	}

	bool lineIntersect(const Vector3 &BeginPoint, const Vector3 &EndPoint) {

		Vector3 Direction = (EndPoint - BeginPoint).normal();
		if ((bxIntersect(BeginPoint, Direction)) && (bxIntersect(EndPoint, -Direction))) {
			return true;
		}

		return false;

	}

	void toString() {

		std::cout << minX << ", " << maxX << ", " << minY << ", " << maxY << ", " << minZ << ", " << maxZ << std::endl;

	}

};

class MeshFace {

public:

	Vector3 v[3];
	Vector3 vt[3];
	Vector3 vn[3];
	Vector3 e1;
	Vector3 e2;
	Vector3 e3;
	Vector3 n;
	Vector3 hn;

	MeshFace() {}
	MeshFace(Vector3 a, Vector3 b, Vector3 c) {

		v[0] = a;
		v[1] = b;
		v[2] = c;

	}

	void addMeshVI(int i, const Vector3 &va) {
		v[i] = va;
	}
	void addMeshNI(int i, const Vector3 &vna) {
		vn[i] = vna;
	}
	void addMeshTI(int i, const Vector3 &vta) {
		vt[i] = vta;
	}

	void initialize() {

		e1 = v[1] - v[0];
		e2 = v[2] - v[0];
		e3 = v[2] - v[1];
		n = e1 * e2;

	}

	//this agrogithm refers to https://blog.csdn.net/silangquan/article/details/17464821
	//and to https://www.cnblogs.com/len3d/archive/2010/07/23/1783365.html
	Vector3 MeshFaceIntersect(const Vector3 &BeginPoint, const Vector3 &Direction) {

		float det = -Direction.dot(n);

		if (det == 0) {
			return Vector3(INFINITY, -1, -1);
		}
		else {

			Vector3 l = BeginPoint - v[0];
			float dett = l.dot(n);
			float t = dett / det;

			if (t<0) {
				return Vector3(INFINITY, -1, -1);
			}

			float detu = -Direction.dot(l*e2);
			float u = detu / det;

			if (u<0 || u>1) {
				return Vector3(INFINITY, -1, -1);
			}

			float detv = -Direction.dot(e1*l);
			float vv = detv / det;

			if (vv<0 || vv>1 || vv + u>1) {
				return Vector3(INFINITY, -1, -1);
			}

			hn = n.normal();
			return Vector3(t, u, vv);

		}
	}

	bool lineIntersect(const Vector3 &BeginPoint, const Vector3 &EndPoint) {

		Vector3 Direction = (EndPoint - BeginPoint).normal();
		return ((MeshFaceIntersect(BeginPoint, Direction).y > -0.1) && (MeshFaceIntersect(EndPoint, -Direction).y > -0.1));

	}

	bool MeshFaceBoxIntersect(BoundingBox &bbBox) {

		//if trangle edge intersect box

		if (
			(bbBox.bspIntersect(v[0], (v[1] - v[0]).normal()) > -BIAS)
			&&
			(bbBox.bspIntersect(v[1], (v[0] - v[1]).normal()) > -BIAS)
			) {
			return true;
		}

		if (
			(bbBox.bspIntersect(v[1], (v[2] - v[1]).normal()) > -BIAS)
			&&
			(bbBox.bspIntersect(v[2], (v[1] - v[2]).normal()) > -BIAS)
			) {
			return true;
		}

		if (
			(bbBox.bspIntersect(v[0], (v[2] - v[0]).normal()) > -BIAS)
			&&
			(bbBox.bspIntersect(v[2], (v[0] - v[2]).normal()) > -BIAS)
			) {
			return true;
		}

		//if bbox edge intersect trangle

		if (lineIntersect(bbBox.p1, bbBox.p2)) {
			return true;
		}

		if (lineIntersect(bbBox.p2, bbBox.p3)) {
			return true;
		}

		if (lineIntersect(bbBox.p3, bbBox.p4)) {
			return true;
		}

		if (lineIntersect(bbBox.p4, bbBox.p1)) {
			return true;
		}

		if (lineIntersect(bbBox.p5, bbBox.p6)) {
			return true;
		}

		if (lineIntersect(bbBox.p6, bbBox.p7)) {
			return true;
		}

		if (lineIntersect(bbBox.p7, bbBox.p8)) {
			return true;
		}

		if (lineIntersect(bbBox.p8, bbBox.p5)) {
			return true;
		}

		if (lineIntersect(bbBox.p1, bbBox.p5)) {
			return true;
		}

		if (lineIntersect(bbBox.p2, bbBox.p6)) {
			return true;
		}

		if (lineIntersect(bbBox.p3, bbBox.p7)) {
			return true;
		}

		if (lineIntersect(bbBox.p4, bbBox.p8)) {
			return true;
		}


		return false;

	}
};

class Phong {

public:
	//SurfaceColor = Emissive + Ambient + Diffuse + Specular
	Vector3 Ke;//Emissive = Ke
	Vector3 Ks;//Specular = Ks * LightColor * pow( max(dot(N, H), 0), fShinines )
	Vector3 Kd;//Diffuse   = Kd * LightColor * max( dot(N, L), 0)
	float Ka; //Ambient = Ka * GlobalAmbient
	float fShiness;
	float Weight_Emissive;
	float Weight_Diffuse;
	float Weight_Reflective;
	float Weight_Transmit;

	float bound0;
	float bound1;
	float bound2;
	float bound3;
	float bound4;

	Phong(const Vector3 &e, const Vector3 &s, const Vector3 &d, float a, float fs, float we = 0, float wd = 0, float wr = 0, float wt = 0) {
		Ke = e;
		Ks = s;
		Kd = d;
		Ka = a;
		fShiness = fs;
		Weight_Emissive = we;
		Weight_Diffuse = wd;
		Weight_Reflective = wr;
		Weight_Transmit = wt;

		bound0 = 0;
		bound1 = bound0 + Weight_Emissive;
		bound2 = bound1 + Weight_Diffuse;
		bound3 = bound2 + Weight_Reflective;
		bound4 = bound3 + Weight_Transmit;

	}

	/*
	int getSurfaceType() {

	//define Emissive = 0;
	//define Diffuse = 1;
	//define reflective = 2;
	//define refractive = 3;

	dice = rand;

	if (dice < bound1) {
	return 0;
	}else if (dice < bound2) {
	return 1;
	}else if (dice < bound3) {
	return 2;
	}else {
	return 3;
	}


	}
	*/

	Phong() {}
	~Phong() {}

};

class Object {

public:
	Vector3 hitPoint;
	Vector3 hitNormal;
	Phong matPhong;

	virtual float Intersect(const Vector3 &BeginPoint, const Vector3 &Direction) {
		std::cout << "bad override" << std::endl;
		return -1;
	}

	virtual float BspIntersect(const Vector3 &BeginPoint, const Vector3 &Direction, std::vector<int> faceInclude) {
		std::cout << "bad bsp override" << std::endl;
		return -1;
	}
	virtual std::vector<int> faceInside(BoundingBox bBox, std::vector<int> faceInclude) {
		std::vector<int> temp;
		return temp;
	}

	virtual int size() {
		std::cout << "bad override" << std::endl;
		return 0;
	}

};

class MeshObj : public Object {

public:

	std::vector<MeshFace> meshInclude;
	BoundingBox bBox;
	bool ifSmooth;

	void toString() {
		std::cout << "Mesh count of this Mesh: " << meshInclude.size() << std::endl;
	}
	void addMeshFace(MeshFace rm) {
		meshInclude.push_back(rm);
	}

	virtual float Intersect(const Vector3 &BeginPoint, const Vector3 &Direction) {

		if (bBox.bspIntersect(BeginPoint, Direction)<-BIAS) {
			return -1;
		}

		int meshk = -1;
		float meshNearest = INFINITY;
		float meshU;
		float meshV;

		for (int i = 0; i < meshInclude.size(); i++) {

			Vector3 testV = meshInclude[i].MeshFaceIntersect(BeginPoint, Direction);

			if (testV.x < meshNearest)
			{
				meshNearest = testV.x;
				meshU = testV.y;
				meshV = testV.z;
				meshk = i;
			}
		}

		if (INFINITY - meshNearest<0.1) {

			return -1;

		}
		else {

			hitPoint = BeginPoint + Direction * meshNearest;

			if (ifSmooth) {

				//Face Normal Smooth
				hitNormal = (meshInclude[meshk].vn[0] * (1 - meshU - meshV) + meshInclude[meshk].vn[1] * (meshU)+meshInclude[meshk].vn[2] * (meshV)).normal();

			}
			else {

				//Face Normal Flat

				hitNormal = meshInclude[meshk].hn;

			}

			return meshNearest;
		}

	}

	virtual std::vector<int> faceInside(BoundingBox bbBox, std::vector<int> faceInclude) {

		std::vector<int> temp;
		std::vector<int>::iterator meshFind;

		for (meshFind = faceInclude.begin(); meshFind != faceInclude.end(); meshFind++) {

			if (meshInclude[*meshFind].MeshFaceBoxIntersect(bbBox)) {
				temp.push_back(*meshFind);
			}

			//std::cout << (*meshFind) << std::endl;

		}

		//std::cout <<faceInclude.size()<< " divide into "<<temp.size() << std::endl;

		return temp;
	}

	virtual float BspIntersect(const Vector3 &BeginPoint, const Vector3 &Direction, std::vector<int> faceInclude) {

		int meshk = -1;
		float meshNearest = INFINITY;
		float meshU;
		float meshV;

		std::vector<int>::iterator meshFind;

		for (meshFind = faceInclude.begin(); meshFind != faceInclude.end(); meshFind++) {

			Vector3 testV = meshInclude[*meshFind].MeshFaceIntersect(BeginPoint, Direction);

			if (testV.x < meshNearest)
			{
				meshNearest = testV.x;
				meshU = testV.y;
				meshV = testV.z;
				meshk = (*meshFind);
			}

		}

		if (INFINITY - meshNearest<0.1) {
			return -1;
		}
		else {
			hitPoint = BeginPoint + Direction * meshNearest;

			if (ifSmooth) {

				//Face Normal Smooth
				hitNormal = (meshInclude[meshk].vn[0] * (1 - meshU - meshV) + meshInclude[meshk].vn[1] * (meshU)+meshInclude[meshk].vn[2] * (meshV)).normal();
			}
			else {
				//Face Normal Flat
				hitNormal = meshInclude[meshk].hn;
			}
			return meshNearest;
		}
		return -1;
	}

	virtual int size() {
		return meshInclude.size();
	}
};

class Camera {

public:

	Vector3 eyePoint;
	Vector3 viewPoint;

	float viewAngle;
	int resolutionU;
	int resolutionV;
	float focus; // focus distance mm

	Vector3 d;
	Vector3 uDir;
	Vector3 vDir;
	float lengthper;

	Camera() {}
	Camera(Vector3 ep, Vector3 vp, float va, int rU, int rV, float f = 18) {
		eyePoint = ep;
		viewPoint = vp;
		viewAngle = va;
		resolutionU = rU;
		resolutionV = rV;
		focus = 18;
	}
	~Camera() {}

	//Camera::initialize() should be called after created

	void initialize() {

		if ((viewPoint - eyePoint).x == 0 && (viewPoint - eyePoint).y == 0) {

			d = (viewPoint - eyePoint).normal();
			uDir = Vector3(1, 0, 0);
			vDir = Vector3(0, 1, 0);

		}
		else {

			d = (viewPoint - eyePoint).normal();
			uDir = Vector3(-d.y, d.x, 0).normal();
			vDir = Vector3(d.x*d.z, d.y*d.z, -d.x*d.x - d.y*d.y).normal();
			lengthper = 2 * tan(M_PI*0.5*viewAngle / 180) / resolutionU;

		}
	}

	Vector3 beginPoint() {
		return eyePoint;
	}

	Vector3 rayDirection(int u, int v) {

		Vector3 Dir = (d + uDir * lengthper*(-resolutionU / 2 + u) + vDir * lengthper*(-resolutionV / 2 + v)).normal();

		return Dir;
	}

};

class SimpleSphere : public Object {

public:

	Vector3 center;
	float radius;


	SimpleSphere(const Vector3 &c, const float &r) {
		center = c;
		radius = r;
	}
	~SimpleSphere() {}

	virtual float Intersect(const Vector3 &BeginPoint, const Vector3 &Direction)
	{

		Vector3 l = center - BeginPoint;
		float tp = l.dot(Direction);
		Vector3 dd = Direction;

		if (tp<0 && l.length()>radius) {
			return -1;

		}
		else {

			float dsquare = l.dot(l) - tp * tp;
			float tvar = (radius*radius - dsquare);

			if (dsquare>radius*radius) {

				return -1;

			}
			else {

				float tt;

				if (l.length()<radius) {
					tt = tp + sqrt(tvar);

				}
				else {
					tt = tp - sqrt(tvar);
				}

				hitPoint = BeginPoint + Direction * tt;
				hitNormal = (hitPoint - center).normal();

				return tt;
			}
		}
	}
	virtual std::vector<int> faceInside(BoundingBox bBox, std::vector<int> faceInclude) {

		std::vector<int> temp;
		if ((center.x + radius > GxMax) || center.x - radius<GxMin) {
			return temp;
		}

		if ((center.y + radius > GyMax) || center.y - radius<GyMin) {
			return temp;
		}

		if ((center.z + radius > GzMax) || center.z - radius<GzMin) {
			return temp;
		}

		temp.push_back(0);
		return temp;
	}

	virtual float BspIntersect(const Vector3 &BeginPoint, const Vector3 &Direction, std::vector<int> faceInclude) {
		std::cout << " bsp work with mesh only!" << std::endl;
		return -1;
	}

	virtual int size() {
		std::cout << "bad override" << std::endl;
		return 1;
	}

};

class SimplePlane : public Object {
public:

	Vector3 planePoint;
	Vector3 planeNormal;

	SimplePlane(Vector3 pp, Vector3 pn = Vector3(0, 0, 1)) {
		planePoint = pp;
		planeNormal = pn.normal();
	}

	virtual float Intersect(const Vector3 &BeginPoint, const Vector3 &Direction)
	{
		float d = -planeNormal.dot(planePoint);
		float t = -(d + planeNormal.dot(BeginPoint)) / (planeNormal.dot(Direction));

		if (t <= 0) {
			return -1;
		}
		else {

			hitPoint = BeginPoint + Direction * t;
			hitNormal = planeNormal;

			return t;
		}
	}

	virtual std::vector<int> faceInside(BoundingBox bBox, std::vector<int> faceInclude) {
		std::vector<int> temp;
		return temp;
	}

	virtual float BspIntersect(const Vector3 &BeginPoint, const Vector3 &Direction, std::vector<int> faceInclude) {
		std::cout << " bsp work with mesh only!" << std::endl;
		return -1;
	}

	virtual int size() {
		std::cout << "bad override" << std::endl;
		return 1;
	}

};

class Scene {

public:

	std::vector<Object*>objIncluded;

	std::vector<SimpleSphere*>SimpleSphereIncluded;
	std::vector<SimplePlane*>SimplePlaneIncluded;
	std::vector<MeshObj*>MeshObjIncluded;

	Vector3 backColor;

	bool ifCopy = false;

	void addObj(SimpleSphere* o) {
		Object *p = o;
		SimpleSphereIncluded.push_back(o);
		objIncluded.push_back(p);
	}

	void addObj(SimplePlane* pl) {
		Object *p = pl;
		SimplePlaneIncluded.push_back(pl);
		objIncluded.push_back(p);
	}

	void addObj(MeshObj* ml) {
		Object *p = ml;
		MeshObjIncluded.push_back(ml);
		objIncluded.push_back(p);
	}
	
	Scene() {}
	//deep copy, for MT
	void SceneCopy(Scene &sc) {

		ifCopy = true;
		backColor = sc.backColor;

		for (int i = 0; i < sc.SimpleSphereIncluded.size(); i++) {
			SimpleSphere *temp = new SimpleSphere(*sc.SimpleSphereIncluded[i]);
			SimpleSphereIncluded.push_back(temp);
			addObj(temp);
		}

		for (int i = 0; i < sc.SimplePlaneIncluded.size(); i++) {
			SimplePlane *temp = new SimplePlane(*sc.SimplePlaneIncluded[i]);
			SimplePlaneIncluded.push_back(temp);
			addObj(temp);
		}

		for (int i = 0; i < sc.MeshObjIncluded.size(); i++) {
			MeshObj *temp = new MeshObj(*sc.MeshObjIncluded[i]);
			MeshObjIncluded.push_back(temp);
			addObj(temp);
		}

	}

	~Scene() {
	
	}

	void toString() {
		std::cout << "This scene contains of " << objIncluded.size() << " objects" << std::endl;
	}
};

/* advanced class and function start, as they need utility instance */

Vector3 resample(const Vector3 &n, const Vector3 &dir) {

	Vector3 resampleU;
	Vector3 resampleV;
	Vector3 var;
	Vector3 result;

	float u;
	float v;
	float radius;
	float theta;
	float temp;

	std::random_device rd;
	std::uniform_int_distribution<int> dist(0, 360);

	std::default_random_engine e;  //生成无符号的随机整数  
	std::uniform_real_distribution<double> ran(0, 1);   //0到1（包含）的均匀分布  

	temp = ran(rd);
	temp = acos(temp);

	//std::cout << temp << std::endl;
	radius = tan(temp);

	theta = dist(rd)*M_PI / 180;

	resampleU = (dir*(n.dot(dir)) - n);
	resampleU = resampleU.normal();
	resampleV = dir * resampleU;

	u = radius * sin(theta);
	v = radius * cos(theta);

	var = resampleU * u + resampleV * v;

	result = dir + var;
	result = result.normal();

	return result;

}

class Photon {

	/*
	this class implement a single Photon used in Photonmapping
	the Photon would generate the map here, not outside the shit
	*/

public:
	Vector3 pose; //position of Photon
	Vector3 radiance; //radiance Red, Bule, Green Channel
	Vector3 dir; //direction of the ray or photon
	bool flag = true;

	Photon(Vector3 _pose, Vector3 _radiance, Vector3 _dir) {

		pose = _pose;
		radiance = _radiance;
		dir = _dir;

	}

	void toString() {

		std::cout << "Photon Position is ";
		pose.toString();

		std::cout << "Photon radiance is ";
		radiance.toString();

		std::cout << "Photon dir is ";
		dir.toString();

	}

	bool Mapping(const Scene &sc, int currentdepth) {

		//return 0, photon is dead, earse it from list
		//return 1, photon is still alive, keep it please

		if (currentdepth > MAX_DEPTH) {
			return false;
		}

		bool intersect;
		int objNear = -1;
		float intersectNear = INFINITY;
		float intersectNearest = INFINITY;

		for (int k = 0; k < sc.objIncluded.size(); k++) {

			intersectNear = sc.objIncluded[k]->Intersect(pose
				, dir);

			if (((intersectNear - intersectNearest)<-0.001) && (intersectNear>-0.1)) {
				intersectNearest = intersectNear;
				objNear = k;
			}
		}

		if (objNear < -0.1)
		{
			//nothing intersect with this shit! the Photon flys out of the scene, so we eliminate it from the list
			return false;

		}
		else {

			Phong objPhong = sc.objIncluded[objNear]->matPhong;
			float bias = BIAS;

			Vector3 realHitPoint = sc.objIncluded[objNear]->hitPoint;
			Vector3 hitNormal = sc.objIncluded[objNear]->hitNormal;
			Vector3 reflect_direction = (dir - hitNormal * 2 * dir.dot(hitNormal)).normal();
			Vector3 hitPoint = sc.objIncluded[objNear]->hitPoint + reflect_direction * bias;

			std::random_device rd;
			std::uniform_real_distribution<double> ran(0, 1);   //0到1（包含）的均匀分布

			float tempMaterial = ran(rd);

			if (

				tempMaterial < objPhong.Weight_Reflective
				) {
				// surface = reflective
				// ref::pathTracing(sc, hitPoint, reflect_direction, depth + 1, returnColor, renderTree);

				pose = hitPoint;
				dir = reflect_direction;

				Mapping(sc, currentdepth + 1);

			}
			else if (
				tempMaterial < objPhong.Weight_Reflective + objPhong.Weight_Transmit
				) {
				//surface = refractive

				Vector3 transmitDir;
				Vector3 transmitHitPoint;
				transmitHitPoint = realHitPoint + dir * bias;

				if (dir.dot(hitNormal)<0) {

					transmitDir = dir - (dir - hitNormal * hitNormal.dot(dir))*0.51;
				}
				else {
					transmitDir = dir + (dir - hitNormal * hitNormal.dot(dir));
				}

				transmitDir = transmitDir.normal();

				pose = transmitHitPoint;
				dir = transmitDir;
				Mapping(sc, currentdepth + 1);

			}
			else {
				//surface = diffuse

				//Vector3(0.98, 0.35, 0.98)

				float absorbR = 1 - objPhong.Kd.x;
				float absorbG = 1 - objPhong.Kd.y;
				float absorbB = 1 - objPhong.Kd.z;

				float absorbAvg = (absorbR + absorbG + absorbB) / 3.0;
				//average absorb odd

				std::random_device rd;
				std::uniform_real_distribution<double> ran(0, 1);   //0到1（包含）的均匀分布

				float tempDecision = ran(rd);

				//followig code is tricky and not easy to understand

				if (tempDecision > absorbAvg) {
					//reflect, not absorb
					//radiance = radiance - (Vector3(1) - objPhong.Kd)*absorbAvg;

					float colorBleeding = ran(rd);

					if (flag && (colorBleeding < absorbAvg)) {

						radiance = radiance - (Vector3(1) - objPhong.Kd);
						flag = false;

					}
					pose = hitPoint;
					dir = resample(hitNormal, dir);

					Mapping(sc, currentdepth + 1);

				}
				else {
					//absorb, not reflect

					if (flag) {
						radiance = radiance - (Vector3(1) - objPhong.Kd);
						flag = false;

					}
					pose = hitPoint;
					dir = resample(hitNormal, dir);
					return true;
				}
			}
		}
	}
};

class PMLight {

public:

	float LIntensity;
	Vector3 center;
	Vector3 UDir;
	Vector3 VDir;

	float ULow;
	float UUp;
	float VLow;
	float VUp;

	Vector3 normal;
	int photonNum;
	std::vector<Photon> photonMap;
	std::vector<Photon*> realPhotonMap;
	std::vector<Photon*> temp[THREAD];
	
	PMLight(float _LIntensity, Vector3 _center, Vector3 _normal, Vector3 _UDir, Vector3 _VDir, float _ULow, float _UUp, float _VLow, float _VUp, int _pNum) {

		LIntensity = _LIntensity;
		center = _center;
		UDir = _UDir;
		VDir = _VDir;

		ULow = _ULow;
		UUp = _UUp;
		VLow = _VLow;
		VUp = _VUp;

		normal = _normal;
		photonNum = _pNum;
		photonMap.reserve(photonNum);

		for (int i = 0; i < THREAD; i++) {
		
			temp[i].reserve((int)(photonNum / THREAD));
		
		}


	}

	void InitialMapping() {

		std::random_device rd;
		std::uniform_int_distribution<int> dist(0, 360);
		std::uniform_real_distribution<double> ran(0, 1);   //0到1（包含）的均匀分布

		std::uniform_real_distribution<double> uRand(ULow, UUp); //randon UV position
		std::uniform_real_distribution<double> vRand(VLow, VUp);

		//add photon to map
		for (int i = 0; i < photonNum; i++) {

			//Calculating Photon Position
			float uShift = uRand(rd);
			float vShift = vRand(rd);
			Vector3 pPosition = center + UDir * uShift + VDir * vShift;

			//Calculating Photon Direction
			float temp = ran(rd);
			temp = acos(temp);

			float radius = tan(temp);
			float theta = dist(rd)*M_PI / 180;

			float uScalar = radius * sin(theta);
			float vScalar = radius * cos(theta);

			Vector3 shotDir = normal + UDir * uScalar + VDir * vScalar;
			shotDir = shotDir.normal();

			//Create Photon and set it in a map
			Photon tempPhoton = Photon(pPosition, Vector3(LIntensity), shotDir);
			photonMap.push_back(tempPhoton);

		}
	}
};

class BspNode {

public:

	BoundingBox BBox;

	BspNode* LChild = nullptr;
	BspNode* RChild = nullptr;
	BspNode* Parent = nullptr;

	//for photon mapping
	std::vector <Photon*> photonIncluded;

	BspNode() {

	}

	int PMsize() {
		return photonIncluded.size();
	}
};

class BspTree {

public:

	int MaxDepth;
	int CurrentDepth;
	BspNode zeroNode;

	BspTree() {}

	//reload for photon mapping
	BspTree(const BoundingBox &gBox, const PMLight &PML, int mD) {

		zeroNode.BBox = gBox;
		zeroNode.photonIncluded = PML.realPhotonMap;

		MaxDepth = mD;
	}

	//reload for photon mapping
	void divideBspTree(BspNode* currentNode, int cD, int cAxielIndex, const PMLight &PML) {

		if (cD >= MaxDepth) {
			return;
		}

		if ((*currentNode).photonIncluded.size() <= 60) {
			return;
		}


		BspNode* cBN_L = new BspNode();
		BspNode* cBN_R = new BspNode();;

		BoundingBox cBox = (*currentNode).BBox;

		//Axiel Index = 0:x -> 1:y
		//Axiel Index = 1:y -> 2:z
		//Axiel Index = 2:z -> 0:x

		if (cAxielIndex == 0) {
			// divide y
			(*cBN_L).BBox = BoundingBox(cBox.minX, cBox.maxX, cBox.minY, (cBox.minY + cBox.maxY) / 2 , cBox.minZ, cBox.maxZ);
			(*cBN_R).BBox = BoundingBox(cBox.minX, cBox.maxX, (cBox.minY + cBox.maxY) / 2 , cBox.maxY, cBox.minZ, cBox.maxZ);

		}
		else if (cAxielIndex == 1) {
			// divide z
			(*cBN_L).BBox = BoundingBox(cBox.minX, cBox.maxX, cBox.minY, cBox.maxY, cBox.minZ, (cBox.minZ + cBox.maxZ) / 2 );
			(*cBN_R).BBox = BoundingBox(cBox.minX, cBox.maxX, cBox.minY, cBox.maxY, (cBox.minZ + cBox.maxZ) / 2 , cBox.maxZ);

		}
		else {
			//divide x
			(*cBN_L).BBox = BoundingBox(cBox.minX, (cBox.minX + cBox.maxX) / 2 , cBox.minY, cBox.maxY, cBox.minZ, cBox.maxZ);
			(*cBN_R).BBox = BoundingBox((cBox.minX + cBox.maxX) / 2 , cBox.maxX, cBox.minY, cBox.maxY, cBox.minZ, cBox.maxZ);

		}

		std::vector<Photon*>::iterator photonIter;

		for (photonIter = (*currentNode).photonIncluded.begin(); photonIter != (*currentNode).photonIncluded.end(); photonIter++) {

			if ((*cBN_L).BBox.ifPointInside(
				(*(*photonIter)).pose
			)) {
				(*cBN_L).photonIncluded.push_back((*photonIter));
			}
			else {
				(*cBN_R).photonIncluded.push_back((*photonIter));
			}
		}

		(*currentNode).LChild = cBN_L;
		(*currentNode).RChild = cBN_R;

		(*cBN_L).Parent = currentNode;
		(*cBN_R).Parent = currentNode;

		(*currentNode).photonIncluded.clear();
		(*currentNode).photonIncluded.swap((*currentNode).photonIncluded);

		//std::string cN, int cD, int cAxielIndex, const PML
		divideBspTree(((*currentNode).LChild), cD + 1, (cAxielIndex + 1) % 3, PML);
		divideBspTree(((*currentNode).RChild), cD + 1, (cAxielIndex + 1) % 3, PML);

		return;
	}

	void traverse(const Vector3 &HitPoint, const PMLight &PML, BspNode* currentNode, float treshold, std::vector<Photon*> &toCheck) {

		//if hitPoint is too far
		if ((*currentNode).BBox.PointDist(HitPoint) > treshold + BIAS) {
			return;
		}

		//if current branch is empty
		if ((*currentNode).photonIncluded.size()<0.5) {

			if ((*currentNode).LChild == nullptr) {
				return;
			}

			//if current empty branch has no child simply return, else dig in another depth
			traverse(HitPoint, PML, (*currentNode).LChild, treshold, toCheck);
			traverse(HitPoint, PML, (*currentNode).RChild, treshold, toCheck);

		}
		else {
			toCheck.insert(toCheck.end(), (*currentNode).photonIncluded.begin(), (*currentNode).photonIncluded.end());
			return;
		}
	}

	int PMsize(BspNode* currentNode) {

		if ((*currentNode).LChild == nullptr) {

			return (*currentNode).PMsize();
		
		}
		else {

			return PMsize((*currentNode).LChild) + PMsize((*currentNode).RChild);
		}
	
	}

	void delNode(BspNode* currentNode) {

		if (currentNode->LChild == nullptr) {
			delete currentNode;
			return;
		}
		else {
			
			delNode(currentNode->LChild);
			delNode(currentNode->RChild);

			if (currentNode != &zeroNode) {
				delete currentNode;
			}
			return;
		}
	}

	~BspTree() {

		delNode(&zeroNode);
	

	
	}

};

void photonMapMapperMT(Scene *sc, const Camera &cam, PMLight* pml, int index, int usingThread) {

	//for each photon inside map, trace

	for (int i = index; i < (*pml).photonMap.size(); i = i + usingThread) {

		if ((*pml).photonMap[i].Mapping(*sc, 0)) {

			float weight = ((*pml).photonMap[i].dir).dot((cam.eyePoint - (*pml).photonMap[i].pose).normal());

			if (weight > 0
				&& (*pml).photonMap[i].radiance.x > 0 && (*pml).photonMap[i].radiance.y>0 && (*pml).photonMap[i].radiance.z >0
				) {
				(*pml).photonMap[i].radiance = (*pml).photonMap[i].radiance*weight;
				(*pml).temp[index].push_back(&((*pml).photonMap[i]));
			}
		}


		if (i % (int)((*pml).photonMap.size() / 20) == 0) {

			std::cout << "photon Mapping generating " << 100 * i / (*pml).photonMap.size() << "% completed" << std::endl;

		}

	}
}

void photonMapGather(PMLight* pml, float treshold, Scene* sc, const Vector3 &beginPoint, const Vector3 &rayDir, int depth, Vector3 &returnColor, BspTree* renderTree) {

	if (depth >= MAX_DEPTH) {
		return;
	}

	bool intersect;
	int objNear = -1;
	float intersectNear = INFINITY;
	float intersectNearest = INFINITY;

	for (int k = 0; k < (*sc).objIncluded.size(); k++) {

		intersectNear = (*sc).objIncluded[k]->Intersect(beginPoint
			, rayDir);

		if (((intersectNear - intersectNearest)<-0.001) && (intersectNear>-0.1)) {
			intersectNearest = intersectNear;
			objNear = k;
		}
	}

	if (objNear < -0.1)
	{
		returnColor = (*sc).backColor;
		return;

	}
	else {

		Phong objPhong = (*sc).objIncluded[objNear]->matPhong;
		float bias = BIAS;

		Vector3 realHitPoint = (*sc).objIncluded[objNear]->hitPoint;
		Vector3 hitNormal = (*sc).objIncluded[objNear]->hitNormal;
		Vector3 reflect_direction = (rayDir - hitNormal * 2 * rayDir.dot(hitNormal)).normal();
		Vector3 hitPoint = (*sc).objIncluded[objNear]->hitPoint + reflect_direction * bias;

		std::random_device rd;
		std::uniform_real_distribution<double> ran(0, 1);   //0到1（包含）的均匀分布

		float tempMaterial = ran(rd);

		if (objPhong.Weight_Emissive>0.1) {
			// surface is emissive
			returnColor = returnColor + objPhong.Ke*std::max(hitNormal.dot(-rayDir), float(0))*objPhong.Weight_Emissive;
			return;
		}
		else if (tempMaterial < objPhong.Weight_Reflective) {
			// surface = reflective
			// ref::pathTracing(sc, hitPoint, reflect_direction, depth + 1, returnColor, renderTree);

			photonMapGather(pml, treshold, sc, hitPoint, reflect_direction, depth + 1, returnColor, renderTree);

		}
		else if (tempMaterial < objPhong.Weight_Reflective + objPhong.Weight_Transmit) {
			//surface = refractive

			Vector3 transmitDir;
			Vector3 transmitHitPoint;
			transmitHitPoint = realHitPoint + rayDir * bias;

			if (rayDir.dot(hitNormal)<0) {

				transmitDir = rayDir - (rayDir - hitNormal * hitNormal.dot(rayDir))*0.51;
			}
			else {
				transmitDir = rayDir + (rayDir - hitNormal * hitNormal.dot(rayDir));
			}

			//ref:: pathTracing(sc, transmitHitPoint, transmitDir, depth + 1, returnColor, renderTree);
			transmitDir = transmitDir.normal();
			photonMapGather(pml, treshold, sc, transmitHitPoint, transmitDir, depth + 1, returnColor, renderTree);

		}
		else {
			//surface = diffuse

			Vector3 gColor = Vector3(0);
			float photonCount = 0;

			//traverse(const Vector3 &HitPoint, PMLight PML, std::string currentBranch, float treshold, std::vector<Photon> &toCheck) {

			std::vector<Photon*> toCheck;
			(*renderTree).traverse(hitPoint, *pml, &(*renderTree).zeroNode, treshold, toCheck);

			//std::cout << toCheck.size() << std::endl;

			if (IFBSPTREE) {

				for (int i = 0; i < toCheck.size(); i++) {

					//for each real photon in PML light, if (distance < treshold) { add to gather}
					//here, we should apply the filter to decrease blur caused by radius

					float dist = (((*toCheck[i]).pose) - hitPoint).length();

					if ( dist < treshold) {
						gColor = gColor + (*toCheck[i]).radiance*(1 - dist/ treshold);
					}
				}

			}
			else {

				for (int i = 0; i < (*pml).realPhotonMap.size(); i++) {

					//for each real photon in PML light, if (distance < treshold) { add to gather} 
					

					if (((
						
						(*(*pml).realPhotonMap[i]).pose
						
						) - hitPoint).length() < treshold) {
						gColor = gColor + (*(*pml).realPhotonMap[i]).radiance;
					}
				}
			}

			returnColor = gColor* LIGHTFACTOR;
			return;

		}
	}
}

void photonMapRenderMT(PMLight* pml, float treshold, Scene* sc, Camera ca, BspTree* renderTree, Mat&draw, int index, int usingThread) {

	for (int i = index; i < ca.resolutionU; i = i + usingThread)
	{
		for (int j = 0; j < ca.resolutionV; j++)
		{
			Vector3 traceResult = Vector3(0);
			photonMapGather(pml, treshold, sc, ca.beginPoint(), ca.rayDirection(i, j), 0, traceResult, renderTree);

			float G = draw.at<Vec3b>(j, i)[0] + (std::max(float(0), (std::min(float(1), traceResult.x)))) * 255;
			float B = draw.at<Vec3b>(j, i)[1] + (std::max(float(0), (std::min(float(1), traceResult.y)))) * 255;
			float R = draw.at<Vec3b>(j, i)[2] + (std::max(float(0), (std::min(float(1), traceResult.z)))) * 255;

			if (G >= 255) {
				draw.at<Vec3b>(j, i)[0] = 255;
			}
			else {
				draw.at<Vec3b>(j, i)[0] = G;
			}

			if (B >= 255) {
				draw.at<Vec3b>(j, i)[1] = 255;
			}
			else {
				draw.at<Vec3b>(j, i)[1] = B;
			}

			if (R >= 255) {
				draw.at<Vec3b>(j, i)[2] = 255;
			}
			else {
				draw.at<Vec3b>(j, i)[2] = R;
			}

		}

		if (index == 0) {
			std::cout << "photon Mapping gathering " << 100 * i / ca.resolutionU << "% completed" << std::endl;
		}

	}
}

/* this function is partially forked from http://www.opengl-tutorial.org/beginners-tutorials/tutorial-7-model-loading/
it set axiel-parellel bounding box automatically */

bool getObjFile(std::string filePath, MeshObj &mO, bool smooth) {

	FILE * file = fopen(filePath.c_str(), "r");

	if (file == NULL) {
		printf("Impossible to open the file !\n");
		return false;
	}
	std::vector<Vector3> vertex;
	std::vector<Vector3> vertexT;
	std::vector<Vector3> vertexN;

	float xMin = INFINITY;
	float xMax = -INFINITY;
	float yMin = INFINITY;
	float yMax = -INFINITY;
	float zMin = INFINITY;
	float zMax = -INFINITY;

	mO.meshInclude;

	while (1) {

		char lineHeader[128];
		// read the first word of the line
		int res = fscanf(file, "%s", lineHeader);
		if (res == EOF) {
			break;
		}
		else {

			if (strcmp(lineHeader, "v") == 0) {

				Vector3 tempVectorVertex;
				fscanf(file, "%f %f %f\n", &tempVectorVertex.x, &tempVectorVertex.y, &tempVectorVertex.z);
				vertex.push_back(tempVectorVertex);

				if (tempVectorVertex.x>xMax) {
					xMax = tempVectorVertex.x;
				}
				if (tempVectorVertex.y>yMax) {
					yMax = tempVectorVertex.y;
				}
				if (tempVectorVertex.z>zMax) {
					zMax = tempVectorVertex.z;
				}
				if (tempVectorVertex.x<xMin) {
					xMin = tempVectorVertex.x;
				}
				if (tempVectorVertex.y<yMin) {
					yMin = tempVectorVertex.y;
				}
				if (tempVectorVertex.z<zMin) {
					zMin = tempVectorVertex.z;
				}


			}
			else if (strcmp(lineHeader, "vt") == 0) {

				Vector3 tempVectorVertexT;
				fscanf(file, "%f %f\n", &tempVectorVertexT.x, &tempVectorVertexT.y);
				vertexT.push_back(tempVectorVertexT);

			}
			else if (strcmp(lineHeader, "vn") == 0) {

				Vector3 tempVectorVertexN;
				fscanf(file, "%f %f %f\n", &tempVectorVertexN.x, &tempVectorVertexN.y, &tempVectorVertexN.z);
				vertexN.push_back(tempVectorVertexN);

			}
			else if (strcmp(lineHeader, "f") == 0) {

				MeshFace readMeshFace;

				std::string vertex1, vertex2, vertex3;
				unsigned int vertexIndex[3], normalIndex[3];
				int matches = fscanf(file, "%d//%d %d//%d %d//%d\n", &vertexIndex[0], &normalIndex[0], &vertexIndex[1], &normalIndex[1], &vertexIndex[2], &normalIndex[2]);

				if (matches != 6) {
					printf("File can't be read by our simple parser : ( Try exporting with other options\n");
					return false;
				}

				readMeshFace.v[0] = vertex[vertexIndex[0] - 1];
				readMeshFace.v[1] = vertex[vertexIndex[1] - 1];
				readMeshFace.v[2] = vertex[vertexIndex[2] - 1];

				readMeshFace.vn[0] = (vertexN[normalIndex[0] - 1]);
				readMeshFace.vn[1] = (vertexN[normalIndex[1] - 1]);
				readMeshFace.vn[2] = (vertexN[normalIndex[2] - 1]);

				readMeshFace.initialize();

				mO.addMeshFace(readMeshFace);
				mO.ifSmooth = smooth;

			}
			else {
			}

		}
	}

	if (xMax>GxMax) {
		GxMax = xMax;
	}
	if (yMax>GyMax) {
		GyMax = yMax;
	}
	if (zMax>GzMax) {
		GzMax = zMax;
	}
	if (xMin<GxMin) {
		GxMin = xMin;
	}
	if (yMin<GyMin) {
		GyMin = yMin;
	}
	if (zMin<GzMin) {
		GzMin = zMin;
	}

	mO.bBox = BoundingBox(xMin - BIAS, xMax + BIAS, yMin - BIAS, yMax + BIAS, zMin - BIAS, zMax + BIAS);
	std::cout << "Mesh object " << filePath << " load successfully" << std::endl;
	return true;
}

int main(void)
{

	BoundingBox GBoundingBox;

	//Scene Setup
	Scene toRender;

	int sWidth = 480;
	int sHeight = 720;

	Mat HW2(sHeight, sWidth, CV_8UC3);

	//Simple Object Setup

	//Mesh Object Setup

	MeshObj SquareLight;
	MeshObj cube;
	MeshObj GreHead;
	MeshObj GlaHead;
	MeshObj GlaBrick;

	MeshObj GlaZoom;
	MeshObj ZoomAss;


	std::string meshpath_7 = "E:\\Notes\\OpenCV_Test\\HW_02_PPM\\greenhead.obj";
	getObjFile(meshpath_7, GreHead, 1);

	std::string meshpath_6 = "E:\\Notes\\OpenCV_Test\\HW_02_PPM\\objCube.obj";
	getObjFile(meshpath_6, cube, 0);

	std::string meshpath_8 = "E:\\Notes\\OpenCV_Test\\HW_02_PPM\\glasshead.obj";
	getObjFile(meshpath_8, GlaHead, 1);

	std::string meshpath_10 = "E:\\Notes\\OpenCV_Test\\HW_02_PPM\\objZoom.obj";
	getObjFile(meshpath_10, GlaZoom, 1);

	std::string meshpath_11 = "E:\\Notes\\OpenCV_Test\\HW_02_PPM\\objZoomAss.obj";
	getObjFile(meshpath_11, ZoomAss, 1);

	GBoundingBox = BoundingBox(GxMin - BIAS, GxMax + BIAS, GyMin - BIAS, GyMax + BIAS, GzMin - BIAS, GzMax + BIAS);

	//Phong Material Setup
	//Phong(const Vector3 &e, const Vector3 &s, const Vector3 &d, float a, float fs, float we = 0, float wd = 0, float wr = 0, float wt = 0) {

	cube.matPhong = Phong(Vector3(0, 0, 0), Vector3(1, 1, 1), Vector3(0.98, 0.35, 0.98), 0.1, 180, 0, 1, 0, 0);
	GreHead.matPhong = Phong(Vector3(0, 0, 0), Vector3(1, 1, 1), Vector3(0.25, 0.65, 0.25), 0.1, 180, 0, 1, 0, 0);
	GlaHead.matPhong = Phong(Vector3(0, 0, 0), Vector3(1, 1, 1), Vector3(0.1, 0.1, 0.1), 0.9, 0.01, 0, 0.1, 0.9, 0.0);
	GlaBrick.matPhong = Phong(Vector3(0, 0, 0), Vector3(1, 1, 1), Vector3(0.1, 0.1, 0.1), 0.1, 10, 0, 0, 0.0, 0.8);
	GlaZoom.matPhong = Phong(Vector3(0, 0, 0), Vector3(0, 0, 0), Vector3(0.0, 0.0, 0.0), 0.1, 10, 0, 0.0, 0.2, 0.8);
	ZoomAss.matPhong = Phong(Vector3(0, 0, 0), Vector3(0, 0, 0), Vector3(0.0, 0.0, 0.0), 0.1, 10, 0, 0.0, 0.0, 1.0);

	//Light setup
	//PointLight(Vector3 cp, Vector3 dc, Vector3 sc = Vector3(1, 1, 1), float diffusePower = 1.0, float specularPower = 1.0){};

	//Camera Setup
	Camera cam = Camera(Vector3(0, -5, 0), Vector3(40, -5, 0), 96, sWidth, sHeight, 18);
	cam.initialize();

	toRender.backColor = Vector3(0);

	toRender.addObj(&GreHead);
	toRender.addObj(&cube);
	toRender.addObj(&GlaZoom);
	toRender.addObj(&GlaHead);
	toRender.addObj(&ZoomAss);
	
	Scene Scene_MT[THREAD];

	for (int i = 0; i < THREAD; i++) {
		Scene_MT[i].SceneCopy(toRender);
	}
		
	if (ENGINE == "PHOTONMAPPING") {

		for (int k = 0; k < BATCHNUM; k++) {
					
			/*PML Generate*/
			std::thread MTMap[THREAD];

			PMLight PMLight_1 = PMLight(1, Vector3(23.650, -3.728, 23.769 - BIAS), Vector3(0, 0, -1), Vector3(1, 0, 0), Vector3(0, 1, 0), -4.9, 4.9, -6.27, 6.27, BATCHSIZE);
			PMLight_1.InitialMapping();
			//std::cout << "before trace, num is: " << PMLight_1.photonMap.size() << std::endl;
						
			for (int i = 0; i < THREAD; i++) {
				MTMap[i] = std::thread(photonMapMapperMT, &Scene_MT[i], cam, &PMLight_1, i, THREAD);
			}

			for (int i = 0; i < THREAD; i++) {
				MTMap[i].join();
			}

			for (int i = 0; i < THREAD; i++) {
				PMLight_1.realPhotonMap.insert(PMLight_1.realPhotonMap.end(), PMLight_1.temp[i].begin(), PMLight_1.temp[i].end());
				PMLight_1.temp[i].clear();
				PMLight_1.temp[i].swap(PMLight_1.temp[i]);
			}
						
			/*BSP Generate*/

			//std::cout << "after trace, num left is: " << PMLight_1.realPhotonMap.size() << std::endl;

			BspTree mainTree(GBoundingBox, PMLight_1, BSPDEPTH);
			mainTree.divideBspTree(&mainTree.zeroNode, 0, 0, PMLight_1);

			//std::cout << "after Tree, num left is: " << mainTree.PMsize(&mainTree.zeroNode) << std::endl;
			
			/*Gather*/
						
			std::thread MT[THREAD];

			for (int i = 0; i < THREAD; i++) {

				MT[i] = std::thread(photonMapRenderMT, &PMLight_1, GATHERRADIUS, &Scene_MT[i], cam, &mainTree, HW2, i, THREAD);
			}

			for (int i = 0; i < THREAD; i++) {
				MT[i].join();
			}
			
			imwrite( std::to_string (k) + "_output.jpg", HW2);
			std::cout << "batch " << k << " completed" << std::endl;
							
		}
	}

	imshow("HW2", HW2);
	imwrite("output.jpg", HW2);

	waitKey(0);

	return 0;
}
