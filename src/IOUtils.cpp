


#include <fstream>
#include <iostream>


#ifdef _WIN32
//#   include "Shellapi.h"
#include <Windows.h>
#	include <direct.h>
#	include <sys/stat.h>
#	define mkdir(a) _mkdir(a)
#	define GetCurrentDir _getcwd
#else
#if !defined(ANDROID)
#   	include <glob.h>
#else
#	include <dirent.h>
#endif
#   include <libgen.h>
#   include <unistd.h>
//#   include <ext/stdio_filebuf.h>
#   include <sys/wait.h>
#	include <sys/stat.h>
#	include <unistd.h>
#	define GetCurrentDir getcwd
#endif

#ifndef S_ISDIR
#define S_ISDIR(mode)  (((mode) & S_IFMT) == S_IFDIR)
#endif

#ifndef S_ISREG
#define S_ISREG(mode)  (((mode) & S_IFMT) == S_IFREG)
#endif

#ifndef MAX_PATH
#   define MAX_PATH 256
#endif

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/types_c.h>
#include "IOUtils.h"
using namespace cv;
using namespace std;

namespace cmp
{

IOUtils::IOUtils(void)
{
}

IOUtils::~IOUtils(void)
{
}


void IOUtils::ShowImageInWindow(Mat img, int flags, const char* windowName)
{
	namedWindow(windowName, flags);
	imshow(windowName, img);
	waitKey();
	cv::destroyWindow(windowName);
}

string IOUtils::SaveTempImage(Mat img, string fileName, const bool forceWrite)
{
#ifdef _DEBUG
	const bool debug = true;
#else
	const bool debug = false;
#endif
	if(forceWrite || debug)
	{
#ifdef _WIN32
		string tempPath = "C:\\Temp\\TextSpotter\\imageOutput\\" + fileName + ".png";
#else
		string tempPath = "/tmp/" + fileName + ".png";
#endif
		imwrite(tempPath, img);
		return tempPath;
	}
	return "";
}


/**
 *
 * @param directory
 * @param searchPattern
 * @param returnFullPath if true, full file path is returned
 * @return files in directory according to search pattern
 */
vector<string> IOUtils::GetFilesInDirectory(const string& directory, const string& searchPattern, bool returnFullPath)
{
	string fullSearch = CombinePath( directory, searchPattern );
#if defined(_WIN32)
	vector<string> files;

	WIN32_FIND_DATA ffd;


	HANDLE hFind = FindFirstFile(fullSearch.c_str(), &ffd);

	if (INVALID_HANDLE_VALUE == hFind)
		return files;



	do
	{
		if (!(ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
		{
			string fileName (ffd.cFileName);
			if( returnFullPath )
				files.push_back(CombinePath(directory, fileName));
			else
				files.push_back(fileName);
		}
	}
	while (FindNextFile(hFind, &ffd) != 0);

	FindClose(hFind);

	return files;
#elif not defined(ANDROID)
	vector<string> files;

	glob_t p;
	glob(fullSearch.c_str(), GLOB_TILDE, NULL, &p);
	for (size_t i=0; i<p.gl_pathc; ++i) {
		if(returnFullPath)
			files.push_back( p.gl_pathv[i] );
		else
			files.push_back( IOUtils::Basename(p.gl_pathv[i]) );

	}
	globfree(&p);

	return files;
#else
	vector<string> files;
	DIR *dir;
	struct dirent *drnt;
	dir = opendir(directory.c_str());
	while ((drnt = readdir(dir)) != NULL)
	{
		string name(drnt->d_name);
		unsigned char type = drnt->d_type;
		if (name != directory && name.length() >= 4)
		{
			if (type == DT_DIR) {
				continue;
			}
			else if (name.find(".png") == (name.length() - 4)) {
				files.push_back( directory + "/" + name );
			}
			else if (name.find(".jpg") == (name.length() - 4)) {
				files.push_back( directory + "/" + name );
			}
		}
	}
	return files;
#endif

}


vector<string> IOUtils::GetDirectoriesInDirectory(const string& directory, const string& searchPattern, bool returnFullPath)
{
	string fullSearch = CombinePath( directory, searchPattern );

#if defined(_WIN32)
	vector<string> directories;

	WIN32_FIND_DATA ffd;


	HANDLE hFind = FindFirstFile(fullSearch.c_str(), &ffd);

	if (INVALID_HANDLE_VALUE == hFind)
		return directories;



	do
	{
		if ((ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
		{
			string fileName (ffd.cFileName);

			if (fileName != "." && fileName != "..")
			{
				if( returnFullPath )
					directories.push_back(CombinePath(directory, fileName));
				else
					directories.push_back(fileName);

			}
		}
	}
	while (FindNextFile(hFind, &ffd) != 0);

	FindClose(hFind);

	return directories;
#elif !defined(ANDROID)

	vector<string> files;

	glob_t p;
	glob(fullSearch.c_str(), GLOB_TILDE, NULL, &p);
	for (size_t i=0; i<p.gl_pathc; ++i) {
		if(returnFullPath)
			files.push_back( p.gl_pathv[i] );
		else
			files.push_back( IOUtils::Basename(p.gl_pathv[i]) );
	}
	globfree(&p);

	return files;
#else
	vector<string> files;
	DIR *dir;
	struct dirent *drnt;
	dir = opendir(directory.c_str());
	while ((drnt = readdir(dir)) != NULL)
	{
		string name(drnt->d_name);
		unsigned char type = drnt->d_type;
		if (name != directory && name.length() >= 4)
		{
			if (type == DT_DIR) {
				continue;
			}
			else if (name.find(".png") == (name.length() - 4)) {
				files.push_back( directory + "/" + name );
			}
			else if (name.find(".jpg") == (name.length() - 4)) {
				files.push_back( directory + "/" + name );
			}
		}
	}
	return files;
#endif
}

bool IOUtils::IsDirectory(const string& path)
{
	bool test = false;
	struct stat stats;
	if (!stat(path.c_str(), &stats)) {
		if (S_ISDIR(stats.st_mode)) {
			test = true;
		}
	}
	return test;
}

/**
 * @param path
 * @return true if path exits on file-system
 */
bool IOUtils::PathExist(const string& path)
{
#ifdef _WIN32
	return ::GetFileAttributes(path.c_str()) != INVALID_FILE_ATTRIBUTES;
#else
	struct stat st;
	if(stat(path.c_str(),&st) == 0)
		return true;

	return false;
#endif
}

std::string IOUtils::CombinePath(std::string directory, std::string file)
{

	string result = directory;
#ifdef _WIN32
	if (result[result.size() -1] != '\\')
		result += '\\';
#else
	if (result[result.size() -1] != '/')
		result += '/';
#endif

	result += file;

	return result;
}

string IOUtils::Basename(string path)
{
#ifdef _WIN32
	string reversed;
	for(string::reverse_iterator c=path.rbegin(); c!=path.rend(); c++)
	{
		if(*c != '\\' && *c!='/' )
		{
			reversed.push_back(*c);
		}
		else break;
	}
	std::reverse(reversed.begin(), reversed.end());
	return reversed;
#else
	char *str = new char[path.size()+1];
	path.copy(str, path.size());
	str[path.size()] = '\0';
	string r=basename(str);
	delete[] str;
	return r;
#endif
}

string IOUtils::RemoveExtension(string str)
{
	return str.substr(0,str.find_last_of("."));
}

string IOUtils::Dirname(string path)
{
#ifdef _WIN32
	cerr << "FIXME: Utils::dirname not implemented on WIN32." << endl;
	return "";
#else
	char *str = new char[path.size()+1];
	path.copy(str, path.size());
	str[path.size()] = '\0';
	string r=dirname(str);
	delete[] str;
	return r;
#endif
}



bool IOUtils::DeleteFile(const char* fileName)
{
#ifdef _WIN32
	return ::DeleteFile(fileName);
#else
	return (unlink(fileName) == 0);
#endif
}

/**
 * Creates new directory
 *
 * No sanity checks!
 * @param dirName
 */
void IOUtils::CreateDir(const std::string& dirName)
{
#if defined(ANDROID)
	cvError(CV_StsError, "Utils::CreateDirectory", "Not implemented!", __FILE__, __LINE__);
#else
	//TODO check results
	mkdir(dirName.c_str(), ALLPERMS);
#endif
}

string IOUtils::GetCurrentDirectory()
{
	char cCurrentPath[FILENAME_MAX];
	if (!GetCurrentDir(cCurrentPath, sizeof(cCurrentPath) / sizeof(char)))
	{
		cv::error(cv::Exception(CV_StsError,  "Utils::GetCurrentDirectory", "Unknown error!", __FILE__, __LINE__));
	}

	string ret = cCurrentPath;
	return ret;
}


string IOUtils::GetFileNameWithoutExtension(string filePath)
{
	int pos1 = filePath.find_last_of('\\');
	int pos2 = filePath.find_last_of('/');
	int pos = max(pos1, pos2);
	string fileNameWithoutExtension = filePath.substr(pos+1);
	fileNameWithoutExtension = fileNameWithoutExtension.substr(0, fileNameWithoutExtension.find_last_of('.'));

	return fileNameWithoutExtension;
}


int IOUtils::StartProcess(string executable, string commandLine)
{
#ifdef _WIN32



	PROCESS_INFORMATION processInformation = {0};
	STARTUPINFO startupInfo                = {0};

	startupInfo.cb                         = sizeof(STARTUPINFO);

	string cmd = executable + " " + commandLine;
	CHAR szCommandLine[MAX_PATH];
	memset(szCommandLine, 0, MAX_PATH);
	strcpy(szCommandLine, cmd.c_str());

	// Create the process
	BOOL result = CreateProcess(NULL, szCommandLine,
			NULL, NULL, TRUE,
			NORMAL_PRIORITY_CLASS,
			NULL, NULL,  &startupInfo, &processInformation);



	if (!result)
		return -1;
	else
		return 0;

#else
	string cmd = executable + " " + commandLine;

	int ret = system(cmd.c_str());
	if (WIFSIGNALED(ret) &&
			(WTERMSIG(ret) == SIGINT || WTERMSIG(ret) == SIGQUIT))
		return -1;
	return 0;
#endif

}

int IOUtils::StartProcessAndWait(string executable, string commandLine, string stdOutputFile)
{
	std::cout << "Running command: " << executable << " with parameters: " << commandLine << std::endl;
#ifdef _WIN32


	PROCESS_INFORMATION processInformation = {0};
	STARTUPINFO startupInfo                = {0};

	startupInfo.cb                         = sizeof(STARTUPINFO);

	HANDLE hOutputFile = INVALID_HANDLE_VALUE;
	if (!stdOutputFile.empty())
	{
		SECURITY_ATTRIBUTES  sec;
		sec.nLength = sizeof(SECURITY_ATTRIBUTES);
		sec.lpSecurityDescriptor = NULL;
		sec.bInheritHandle = TRUE;

		hOutputFile = CreateFile ( stdOutputFile.c_str(),
				GENERIC_WRITE,
				FILE_SHARE_READ | FILE_SHARE_WRITE,
				&sec,
				CREATE_ALWAYS,
				FILE_ATTRIBUTE_NORMAL,
				NULL);

		if (hOutputFile != INVALID_HANDLE_VALUE)
		{
			startupInfo.dwFlags = STARTF_USESHOWWINDOW | STARTF_USESTDHANDLES;
			startupInfo.wShowWindow  =   SW_HIDE;
			startupInfo.hStdOutput = hOutputFile;


		}
	}




	string cmd = executable + " " + commandLine;
	CHAR szCommandLine[MAX_PATH];
	memset(szCommandLine, 0, MAX_PATH);
	strcpy(szCommandLine, cmd.c_str());

	// Create the process
	BOOL result = CreateProcess(NULL, szCommandLine,
			NULL, NULL, TRUE,
			NORMAL_PRIORITY_CLASS,
			GetEnvironmentStrings(), NULL, &startupInfo, &processInformation);



	if (!result)
		return -1;

	// Successfully created the process.  Wait for it to finish.
	WaitForSingleObject( processInformation.hProcess, INFINITE );

	// Get the exit code.
	DWORD exitCode;
	result = GetExitCodeProcess(processInformation.hProcess, &exitCode);

	// Close the handles.
	CloseHandle( processInformation.hProcess );
	CloseHandle( processInformation.hThread );

	if (hOutputFile != INVALID_HANDLE_VALUE)
		CloseHandle(hOutputFile);

	if (!result)
	{
		// Could not get exit code.
		return -2;
	}

	return (int)exitCode;


#else
	string cmd = executable + " " + commandLine;
	if (!stdOutputFile.empty())	{
		cmd += " > " + stdOutputFile;
	}
	int ret = system(cmd.c_str());
	if (WIFSIGNALED(ret) &&
			(WTERMSIG(ret) == SIGINT || WTERMSIG(ret) == SIGQUIT))
		return -1;
	return 0;
#endif

}

std::string IOUtils::RemoveBasepath(string pathstr, int level)
{
#ifdef _WIN32
	char separator='\\';
#else
	char separator='/';
#endif
	int pos=0;
	for(string::iterator c=pathstr.begin(); c!=pathstr.end(); ++c)
	{
		if (level==0) break;

		if(*c==separator)
		{
			level--;
		}
		pos++;
	}
	return pathstr.substr(pos);
}

std::string IOUtils::GetTempPath(void)
{
#ifdef _WIN32

	TCHAR lpTempPathBuffer[MAX_PATH];
	::GetTempPath(MAX_PATH,  lpTempPathBuffer);

	return lpTempPathBuffer;
#else
	return ("");	//TODO: Linux version
#endif
}

void IOUtils::CpFile(const std::string& source, const std::string& dst)
{
	std::ifstream src( source.c_str(), ios::binary );
	ofstream dest( dst.c_str(), ios::binary);

	dest << src.rdbuf();

	src.close();
	dest.close();
}

}//namespace cmp

