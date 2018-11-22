#include <iostream> 
#include <fstream>
#include <sstream>
#include <string>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

VectorXd load_pgm(const std::string &filename) {
	// returns a picture as a flattened vector

	int row = 0, col = 0, rows = 0, cols = 0;

	std::ifstream infile(filename);
	std::stringstream ss;
	std::string inputLine = "";

	// First line : version
	std::getline(infile,inputLine);

	// Second line : comment
	std::getline(infile,inputLine);

	// Continue with a stringstream
	ss << infile.rdbuf();
	// Third line : size
	ss >> cols >> rows;

	VectorXd picture(rows*cols);

	// Following lines : data
	for(row = 0; row < rows; ++row) {
		for (col = 0; col < cols; ++col) {
			int val;
			ss >> val;
			picture(col*rows + row) = val;
		}
	}

	infile.close();
	return picture;
}

int main() {
	
	int h = 231;
	int w = 195;
	int M = 15;

	MatrixXd faces(h*w, M);
	VectorXd meanFace(h*w);
        MatrixXd A(h*w, M);
	
	// loads pictures as flattened vectors into faces
	for (int i=0; i<M; i++) {
		std::string filename = "./basePictures/subject"+ 
			std::to_string(i+1) + ".pgm";
		VectorXd flatPic = load_pgm(filename);
		faces.col(i) = flatPic;
                A.col(i) = flatPic;
		
                meanFace += flatPic;
	}

        meanFace /= M;

        // Subtract mean from A
        A.colwise() -= meanFace;

	
	// TODO: Point (e)
        JacobiSVD<MatrixXd> svd(A, ComputeThinU | ComputeThinV);
        MatrixXd U = svd.matrixU();

	// try to recognize a test face
	string testPicName = "./testPictures/subject01.happy.pgm";
	VectorXd newFace = load_pgm(testPicName);

	// TODO: Point (f)
        VectorXd projNewFace = U.transpose() * (newFace - meanFace);
	
	// TODO: Point (g)
        MatrixXd projFaces = U.transpose() * A;
	int indexMinNorm;
	(projFaces.colwise()-projNewFace).colwise().norm().minCoeff(&indexMinNorm);
	cout << testPicName << " is identified as subject "
		 << indexMinNorm+1 << endl;
}
