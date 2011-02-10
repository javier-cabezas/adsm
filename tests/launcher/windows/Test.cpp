#include <windows.h>

#include "../Test.h"
    
void
Test::TestCase::run(std::string exec)
{
    /* CreateProcess API initialization */ 
    STARTUPINFO startupInfo; 
    PROCESS_INFORMATION processInfo; 
    ::memset(&startupInfo, 0, sizeof(startupInfo)); 
    ::memset(&processInfo, 0, sizeof(processInfo)); 
    startupInfo.cb = sizeof(startupInfo); 

    setEnvironment();
    printf("Launching: %s\n", exec.c_str());
    TCHAR tmpCmdLine[MAX_PATH * 2];
    const char *appName = exec.c_str();
    ::memcpy(tmpCmdLine, appName, strlen(appName) + 1);
    BOOL created = ::CreateProcess(NULL, tmpCmdLine, NULL, NULL, FALSE,
        0, NULL, NULL, &startupInfo, &processInfo);
    if (created) {
        // Wait until child processes exit.
        WaitForSingleObject(processInfo.hProcess, INFINITE);
        CloseHandle(processInfo.hProcess);
        CloseHandle(processInfo.hThread);
    } else {
        printf("Failure! execve error code %d\n", created);
        abort(); 
    }
}

void
Test::TestCase::setEnvironment()
{
    std::vector<KeyVal>::const_iterator it;
    for (it = keyvals_.begin(); it != keyvals_.end(); it++) {
        ::_putenv((it->first + "=" + it->second).c_str());
    }
    ::_putenv("PATH=.");
}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
