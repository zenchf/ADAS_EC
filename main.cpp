// 	g++ -std=c++17 -Wall -o bsur bsur.cpp -ljetgpio `pkg-config --cflags --libs opencv4` -pthread
// ./darknet detector demo ./lyec13/lyec13.data ./lyec13/lyec13.cfg ./lyec13/lyec13b.weights -c 2 -json_port 8888 -thresh 0.25

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <boost/asio.hpp>
#include "/home/nvidia/Documents/pybind11/son_hali/lib_adas25/async-tcp/async-sockets/include/tcpsocket.hpp"
#include "/home/nvidia/Documents/pybind11/son_hali/lib_adas25/struct_mapping/include/struct_mapping/struct_mapping.h"
#include <fstream>
#include <chrono>
#include <algorithm>
#include <list>
#include <vector>
#include <functional>
#include <memory>
#include <iterator>
#include <sstream>
#include <unistd.h>
#include <jetgpio.h>

using namespace cv;
using namespace std;
using boost::asio::ip::tcp;
using namespace struct_mapping;
using namespace std::chrono_literals;

    // Camera setup
    VideoCapture cap_serit(4, CAP_V4L2);
    VideoCapture cap_arka(0);

// GPIO pins
const int led = 8;
const int anahtar = 10;
string tabela = "bsur";

float x, y, w, h;

std::thread tcpThread;
std::atomic<bool> tcpRunning(true);


//--------traffic sign detector reading json
	float signSize, signConfidence;
	int signCount, signStart, signEnd, nextSign, lastSignSize, nextStart = 0;
	string received, selectedSign, obstacle, signData;
	string objects = ""; 
	ofstream outFile;

struct relative_coordinates {
	float center_x;
	float center_y;
	float width;
	float height;
};

struct structObjects {
	string name;
	int class_id;
	float confidence;
	relative_coordinates coords;
};

structObjects jsonObjects;

inline bool ends_with(std::string const & value, std::string const & ending) {
	if (ending.size() > value.size()) return false;
	return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}
//---------------------------------------------------------------
TCPSocket tcpSocket([](int errorCode, std::string errorMessage) {
	cout << "Socket creation error:" << errorCode << " : " << errorMessage << endl;
});

void tcpSetup() {
	tcpSocket.onMessageReceived = [](string message) {
		received = message;	
	};

	tcpSocket.onSocketClosed = [](int errorCode) {
		cout << "Connection closed: " << errorCode << endl;
	};

	tcpSocket.Connect("localhost", 8888, [&] {
		cout << "Connected to the server successfully!" << endl;
		tcpSocket.Send("Hello Server!"); 
	}, [](int errorCode, std::string errorMessage) {
		cout << "HATA: " << errorCode << " : " << errorMessage << endl;
	});
	}
	
void count() {
	signCount = 0;
	signSize = 0;
	signData = "";
	objects = "";
	signStart = 0;
	lastSignSize = 0;
	signEnd = 0;
	nextStart = 0;
	objects = received.substr(received.find('[') + 5, received.find(']'));
	cout << "Parsing: " << received << endl << endl;

	if(!ends_with(objects, "}")) objects += "}";

	do {
		cout << "Parsing new sign." << endl;
		nextSign = objects.find("class_id", nextSign) + 10;
		signStart = objects.find('{', nextStart);
		signEnd = objects.find('}', objects.find("confidence") + 1);
		signData = objects.substr(signStart, signEnd);

		if(!ends_with(signData, "}")) signData += "}";

		if(signData.rfind(':') == signData.length() - 2)
			signData.insert(signData.rfind(':') + 1, "0.0");

		else {
			if(signData.rfind('"') == signData.length() - 2)
				signData.insert(signData.rfind('"') + 1, ":0.0");

			if(signData.rfind('e') == signData.length() - 2)
				signData.insert(signData.rfind('e') + 1, R"(":0.0)");
		}

		if(received.find("{") == 0) {
			istringstream json_data(signData);
			map_json_to_struct(jsonObjects, json_data);
		}

		ostringstream out_json_data;
		map_struct_to_json(jsonObjects, out_json_data, "  ");
		cout << endl << "Sign JSON: " << endl << out_json_data.str() << endl;

		signConfidence = jsonObjects.confidence * 100;
		signSize = (jsonObjects.coords.height * 100) * (jsonObjects.coords.width * 100);

		if(signSize >= lastSignSize) {
			selectedSign = signData;
			lastSignSize = signSize;
		}

		signCount++;
		nextStart += 80;
	} while(objects.find("class_id", nextSign) < objects.length());

	nextSign = 0;
	cout << endl << "Selected Sign: " << selectedSign << endl;

	istringstream json_data(selectedSign);
	map_json_to_struct(jsonObjects, json_data);
	
	x = jsonObjects.coords.center_x;
	y = jsonObjects.coords.center_y;
	w = jsonObjects.coords.width ;
	h = jsonObjects.coords.height ;
	

	cout << "Sign Name: " << jsonObjects.name << endl;
	cout << "Class ID: " << jsonObjects.class_id << endl;
	cout << "Confidence: " << jsonObjects.confidence << endl;
	cout << "Center X: " << jsonObjects.coords.center_x << endl;
	cout << "Center Y: " << jsonObjects.coords.center_y << endl;
	cout << "Width: " << jsonObjects.coords.width << endl;
	cout << "Height: " << jsonObjects.coords.height << endl;
	cout << "Sign Size: " << signSize << endl;
	cout << "Sign Confidence: " << signConfidence << endl;
	cout << "Sign Count: " << signCount << endl;	
	}
	
void detectAndProcess() {
	if (received.size() > 100 && received.find("") == 0 && received != "," && received.rfind("}") == received.size() - 1) {
		count();
	} else {
		jsonObjects.name = "NULL";
		jsonObjects.class_id = -1;
		jsonObjects.confidence = 0;
		jsonObjects.coords.center_x = 0;
		jsonObjects.coords.center_y = 0;
		jsonObjects.coords.height = 0;
		jsonObjects.coords.width = 0;
		signConfidence = 0;
		signSize = 0;
		signCount = 0;
	}
}

string signInfo() {
    int max_attempts = 50;
    while (max_attempts--) {
        try {
            detectAndProcess();

            // Check if we have valid data
            if (jsonObjects.name != "NULL" && !jsonObjects.name.empty()) {
                return jsonObjects.name;
            }

            // Additional safety checks
            if (received.empty() || received.size() < 10) {
                std::this_thread::sleep_for(200ms);
                continue;
            }

            // Find the JSON array in the received string
            size_t array_start = received.find('[');
            size_t array_end = received.find(']');

            // Validate array positions
            if (array_start == string::npos || array_end == string::npos || 
                array_start >= array_end || array_end > received.size()) {
                std::this_thread::sleep_for(200ms);
                continue;
            }

            // Extract objects string with bounds checking
            if (array_start + 5 < received.size()) {
                objects = received.substr(array_start + 1, array_end - array_start - 1);
            } else {
                objects = "";
            }

            // Rest of your processing logic...
            // Make sure to add similar bounds checking for all string operations

        } catch (const std::exception& e) {
            std::cerr << "Error processing sign info: " << e.what() << std::endl;
            jsonObjects.name = "NULL";
            jsonObjects.class_id = -1;
            // Reset other fields as needed
        }

        std::this_thread::sleep_for(200ms);
    }

    return "bsur";
}

//------ Image processing functions ------
Mat getROI(const Mat& frame, int roiHeight) {
    if (frame.empty() || roiHeight <= 0 || roiHeight > frame.rows) {
        return Mat();
    }
    Rect roi(0, frame.rows - roiHeight, frame.cols, roiHeight);
    return frame(roi);
}

pair<Mat, Mat> splitROI(const Mat& frame, float width) {
    if (frame.empty() || width <= 0) {
        return {Mat(), Mat()};
    }
    
    float mid = width / 4;
    Mat leftROI = frame(Rect(0, 0, mid, frame.rows));
    Mat rightROI = frame(Rect((mid*3), 0, mid, frame.rows));
    return {leftROI, rightROI};    
}

Mat preprocessROI(const Mat& roi) {
    if (roi.empty()) return Mat();
    
    Mat blurred, gray, binary;
    GaussianBlur(roi, blurred, Size(7,7), 0);
    cvtColor(blurred, gray, COLOR_BGR2GRAY);
    //threshold(gray, binary, 0, 255, THRESH_OTSU);
    return gray;
}

Point getMaxWhiteX(const Mat& roi) {
    if (roi.empty()) return Point(0, 0);
    
    Mat hist;
    reduce(roi, hist, 0, REDUCE_SUM, CV_32F);
    Point MaxLoc;
    minMaxLoc(hist, 0, 0, 0, &MaxLoc);
    return MaxLoc;
}

int drawVectors(Mat& frame, const Point& leftPt, const Point& rightPt, int lineLen, float angle) {
    if (frame.empty()) return 100;  // Default center position
    
    int skewAmount = angle;
    Point leftVecEnd(leftPt.x + skewAmount, leftPt.y - lineLen);
    Point rightVecEnd(rightPt.x - skewAmount, rightPt.y - lineLen);
    
    line(frame, leftPt, leftVecEnd, Scalar(255, 0, 0), 10); 
    line(frame, rightPt, rightVecEnd, Scalar(0, 0, 255), 10);
    
    Point middlePt((leftPt.x + rightPt.x) / 2, (leftPt.y + rightPt.y) / 2);
    Point middleVecEnd(middlePt.x, middlePt.y - lineLen);
    line(frame, middlePt, middleVecEnd, Scalar(0, 255, 0), 10);
    
    Point bottomCenter(frame.cols / 2, frame.rows);
    Point refAligned(middlePt.x, bottomCenter.y - 30);
    line(frame, bottomCenter, refAligned, Scalar(0, 255, 255), 7);

    int offsetX = middlePt.x - bottomCenter.x;
    int steering = 100 + (offsetX * 100) / (frame.cols / 2);
    steering = std::clamp(steering, 0, 200);

    putText(frame, "Direksiyon: " + to_string(steering), Point(50, 50), 
            FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
    
    return steering;
}

Mat laneWarn(Mat frame, int roiHeight, int minThresh, int maxThresh, int ledpin) {
    
    gpioSetMode(ledpin, JET_OUTPUT);
    
    if (frame.empty()) return frame;
    
    Mat roi = getROI(frame, roiHeight);
    if (roi.empty()) return frame;
    
    Mat binary = preprocessROI(roi);
    auto [leftROI, rightROI] = splitROI(binary, frame.cols);
    
    /*if (!binary.empty()) {
        imshow("Binary ROI", binary);
    }*/
    
    Point leftMax = getMaxWhiteX(leftROI);
    Point rightMax = getMaxWhiteX(rightROI);
    
    Point leftPt(leftMax.x, frame.rows - roiHeight / 2);
    Point rightPt(rightMax.x + (frame.cols * 3 / 4), frame.rows - roiHeight / 2);
    
    int steering = drawVectors(frame, leftPt, rightPt, 40, 80);
    
    // Lane departure warning
    bool warning = (steering < minThresh || steering > maxThresh);
    if (warning) {
        putText(frame, "SERITTEN CIKMAYIN!", Point(100, 100), 
                FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 3);
        gpioWrite(ledpin, 1); //yak
    }
    else
    	gpioWrite(ledpin, 0); //sondur
    	
    
    return frame;
}

Mat GUI(Mat serit, string sign = "bsur", string seritDurumu = "Şeritte", string arkaKameraDurumu = "Kapalı") {
    int genislik = 1920;
    int yukseklik = 1080;
    Mat pencere(yukseklik, genislik, CV_8UC3, Scalar(30, 30, 30)); // Koyu gri arka plan

    // --- ÜST BANNER ---
    rectangle(pencere, Point(0, 0), Point(genislik, 100), Scalar(50, 50, 50), FILLED);
    putText(pencere, "ADAS SISTEM ARAYUZU", Point(600, 65),
            FONT_HERSHEY_SIMPLEX, 2, Scalar(255, 255, 255), 4, LINE_AA);
    
    // --- SOL ÜST: Trafik Tabelası ---
    rectangle(pencere, Point(40, 120), Point(500, 600), Scalar(70, 70, 90), FILLED);
    putText(pencere, "Trafik Tabelasi", Point(70, 170),
            FONT_HERSHEY_SIMPLEX, 1.4, Scalar(255, 255, 255), 2);

    // Tabela resmi
    Mat tabela;
    string yol = "/home/nvidia/Documents/pybind11/son_hali/lib_adas25/tabelalar/" + sign + ".png";
    tabela = imread(yol);
    if (tabela.empty()) {
        tabela = Mat(450, 450, CV_8UC3, Scalar(100, 100, 100));
        putText(tabela, "TABELA YOK", Point(100, 225), 
                FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 0, 255), 3);
    }
    resize(tabela, tabela, Size(420, 420));
    tabela.copyTo(pencere(Rect(70, 180, tabela.cols, tabela.rows)));

    // --- SAĞ ÜST: Tabela Bilgisi Alanı (boş bırakıldı, istenirse doldurulur) ---
    rectangle(pencere, Point(1050, 120), Point(1850, 600), Scalar(60, 60, 80), FILLED);
    putText(pencere, "Tabela Bilgileri", Point(1300, 170),
            FONT_HERSHEY_SIMPLEX, 1.2, Scalar(200, 200, 255), 2);

    // --- SOL ALT: Şerit Takibi ---
    rectangle(pencere, Point(40, 610), Point(980, 1070), Scalar(70, 70, 90), FILLED);
    putText(pencere, "Serit Takibi", Point(60, 660),
            FONT_HERSHEY_SIMPLEX, 1.4, Scalar(255, 255, 255), 2);

    if (!serit.empty()) {
        Mat serit_kucultulmus;
        resize(serit, serit_kucultulmus, Size(900, 400));
        serit_kucultulmus.copyTo(pencere(Rect(60, 680, serit_kucultulmus.cols, serit_kucultulmus.rows)));
    } else {
        putText(pencere, "Veri Yok", Point(300, 850), 
                FONT_HERSHEY_SIMPLEX, 1.2, Scalar(200, 200, 200), 2);
    }

    // --- SAĞ ALT: Navigasyon Haritası + Nokta ---
    rectangle(pencere, Point(1050, 610), Point(1850, 1070), Scalar(70, 70, 90), FILLED);
    putText(pencere, "Navigasyon Haritasi", Point(1080, 660),
            FONT_HERSHEY_SIMPLEX, 1.4, Scalar(255, 255, 255), 2);
    
    Mat harita = imread("/home/nvidia/Documents/pybind11/son_hali/lib_adas25/tabelalar/harita.jpeg");
    if (harita.empty()) {
        harita = Mat(400, 800, CV_8UC3, Scalar(80, 80, 80));
        putText(harita, "HARITA YOK", Point(150, 200), 
                FONT_HERSHEY_SIMPLEX, 1.5, Scalar(255, 255, 0), 3);
    }
    resize(harita, harita, Size(780, 400));

    // Harita üzerine kullanıcı konumunu gösteren mavi nokta
    circle(harita, Point(390, 200), 10, Scalar(255, 0, 0), FILLED); // Mavi nokta
    putText(harita, "konum", Point(405, 190), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 2);

    harita.copyTo(pencere(Rect(1060, 680, harita.cols, harita.rows)));

    return pencere;
}
void tcpWorker() {
    tcpSetup();  // TCP bağlantısını yapar

    while (tcpRunning) {
    	
        if (received.size() > 100 && received.find('{') != std::string::npos) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1350));
            tabela = signInfo();
            received = "";
        }
    }
}

/*
Mat GUI(Mat serit, string sign = "dur", string laneState = "Şeritte", string backupCam = "Kapalı") {
    int width = 1920;
    Mat pencere(height, width, CV_8UC3, Scalar(0, 0, 0));

    // --- ÜST KIRMIZI BANNER ---
    rectangle(pencere, Point(0, 0), Point(width, 100), Scalar(0, 0, 255), FILLED);
    putText(pencere, "ADAS FONKSIYONLARI", Point(width / 2 - 300, 65),
            FONT_HERSHEY_SIMPLEX, 2, Scalar(255, 255, 255), 4, LINE_AA);
    
    // --- SOL ÜST: Trafik Tabelasi ---
    rectangle(pencere, Point(50, 120), Point(480, 600), Scalar(255, 0, 0), FILLED);
    putText(pencere, "Trafik Tabelasi", Point(80, 170),
            FONT_HERSHEY_SIMPLEX, 1.2, Scalar(255, 255, 255), 2);

    // Tabela resmi
    Mat tabela;
    string path = "/home/nvidia/Documents/pybind11/son_hali/lib_adas25/tabelalar/" + sign + ".png";
    tabela = imread(path);
    if (tabela.empty()) {
        tabela = Mat(450, 450, CV_8UC3, Scalar(100, 100, 100));
        putText(tabela, "NO SIGN", Point(100, 225), 
                FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 0, 255), 3);
    }
    resize(tabela, tabela, Size(450, 450));
    tabela.copyTo(pencere(Rect(150, 130, tabela.cols, tabela.rows)));

    // --- SAĞ ÜST: Tabela Bilgisi ---
    rectangle(pencere, Point(1010, 120), Point(1870, 600), Scalar(255, 0, 0), FILLED);

    // --- SOL ALT: Şerit Takibi ---
    rectangle(pencere, Point(50, 610), Point(980, 1070), Scalar(255, 0, 0), FILLED);
    putText(pencere, "Serit Takibi", Point(80, 660),
            FONT_HERSHEY_SIMPLEX, 1.2, Scalar(255, 255, 255), 2);

    if (!serit.empty()) {
        Mat serit_resized;
        resize(serit, serit_resized, Size(900, 400));
        serit_resized.copyTo(pencere(Rect(60, 680, serit_resized.cols, serit_resized.rows)));
    } else {
        putText(pencere, "No Lane Data", Point(300, 850), 
                FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
    }

    // --- SAĞ ALT: Navigasyon ---
    rectangle(pencere, Point(1010, 610), Point(1870, 1070), Scalar(255, 0, 0), FILLED);
    putText(pencere, "Navigasyon", Point(1040, 660),
            FONT_HERSHEY_SIMPLEX, 1.2, Scalar(255, 255, 255), 2);
    
    Mat harita = imread("/home/nvidia/Documents/pybind11/son_hali/lib_adas25/tabelalar/harita.jpeg");
    if (harita.empty()) {
        harita = Mat(400, 800, CV_8UC3, Scalar(100, 100, 100));
        putText(harita, "NAVIGATION MAP", Point(150, 200), 
                FONT_HERSHEY_SIMPLEX, 1.5, Scalar(255, 255, 0), 3);
    }
    resize(harita, harita, Size(800, 400));
    harita.copyTo(pencere(Rect(1060, 680, harita.cols, harita.rows)));

    return pencere;
}*/
int main(int argc, char *argv[]) {
    // TCP verisini dinleyen thread'i başlat
   tcpThread = std::thread(tcpWorker);  // <--- BU YENİ

    // JSON mapping kayıtları
    reg(&relative_coordinates::center_x, "center_x");
    reg(&relative_coordinates::center_y, "center_y");
    reg(&relative_coordinates::width, "width");
    reg(&relative_coordinates::height, "height");

    reg(&structObjects::class_id, "class_id");
    reg(&structObjects::name, "name");
    reg(&structObjects::confidence, "confidence");
    reg(&structObjects::coords, "relative_coordinates");

    // GPIO initialization
    int Init = gpioInitialise();
    if (Init < 0) {
        cerr << "Jetgpio initialization failed. Error code: " << Init << endl;
        return Init;
    }
    cout << "Jetgpio initialized successfully." << endl;
    gpioSetMode(anahtar, JET_INPUT);

    if (!cap_serit.isOpened()) {
        cerr << "Şerit kamerası açılamadı!" << endl;
    }
    if (!cap_arka.isOpened()) {
        cerr << "Arka Kamera acilamadi!" << endl;
    }
    bool arka_cam;
    Mat arka_;
    Mat serit_;
    //int frameCounter = 0;
    
    namedWindow("Arka Kamera", WINDOW_NORMAL);
    resizeWindow("Arka Kamera", 650, 460);
    
    while (true) {
        // Kamera verisi al
        cap_serit >> arka_;
        cap_arka >> serit_;
	
        Mat laneFrame;
        if (!arka_.empty()) {
            laneFrame = laneWarn(arka_.clone(), 100, 80, 120, led);
        } else {
            laneFrame = Mat(480, 640, CV_8UC3, Scalar(0,0,0));
            putText(laneFrame, "Kamera gosterişemiyor", Point(100, 240), 
                    FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0,0,255), 3);
        }
        int level = gpioRead(anahtar);
        /*cout << x<<" "<< y <<" "<< " "<<w <<" " << h << endl;
        Rect rect(x*650, y*460, w*650, h*460);
            rectangle(serit_,rect, Scalar(0, 0, 255), 7 );
            cout << "Arka Kamera acil" << endl;
            imshow("Arka Kamera", serit_);*/
            
        cout << "Switch durumu: " << level << endl;
        if (level && !serit_.empty()) {
            arka_cam = true;
            Rect rect(x, y, w, h);
            rectangle(serit_,rect, Scalar(0, 0, 255), 7 );
            cout << "Arka Kamera acil" << endl;
            imshow("Arka Kamera", serit_);
        } else {
            cout << "Arka Kamera kapali" << endl;
            if(arka_cam){
            destroyWindow("Arka Kamera");
            arka_cam = false;
            }
        }
        
	 cout << "TABELA:" <<tabela<< endl;
            received = "";

        // GUI oluştur
        Mat gui = GUI(laneFrame, tabela);
        imshow("ADAS System", gui);
        // ESC ile çık
        if (waitKey(30) == 27) break;
    }

    tcpRunning = false;
    if (tcpThread.joinable()) tcpThread.join(); 
    cap_serit.release();
    gpioTerminate();
    destroyAllWindows();

    return 0;
}

