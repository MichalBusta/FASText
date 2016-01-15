#pragma once

#include <opencv2/core/core.hpp>

#include <string>
#include <vector>

namespace cmp
{
	/**	 
	
	 @brief	Input/Output utility methods. 
	
	 @author	Lukas Neumann <neumalu1@cmp.felk.cvut.cz>
	 @date	3.9.2012

	 */
	class IOUtils
	{
	private:
		IOUtils(void);
		~IOUtils(void);

	public:

		static std::vector<std::string> GetFilesInDirectory( const std::string& directory, const std::string& searchPattern, bool returnFullPath = false );
		static std::vector<std::string> GetDirectoriesInDirectory(const std::string& directory, const std::string& searchPattern, bool returnFullPath = false);

		static std::string GetFileNameWithoutExtension(std::string filePath);
		static std::string RemoveBasepath(std::string str, int level=1);
		static std::string CombinePath(std::string directory, std::string file);
		static std::string Basename(std::string path);
		static std::string Dirname(std::string path);
		static std::string RemoveExtension(std::string str);

		static std::string GetTempPath();

		static bool DeleteFile(const char* fileName);
		static void CreateDir(const std::string& dirName);

		static std::string GetCurrentDirectory();
		static bool IsDirectory(const std::string& path);
		static bool PathExist(const std::string& path);

		static int StartProcessAndWait(std::string executable, std::string commandLine, std::string stdOutputFile);
		static int StartProcess(std::string executable, std::string commandLine);

		static void ShowImageInWindow(cv::Mat img, int flags = 1, const char* windowName = "Image");
        static std::string SaveTempImage(cv::Mat img, std::string windowName, const bool forceWrite=false);

        static void CpFile( const std::string& source, const std::string& dst );
	};

}
