"""
Find what is the available memory.
"""

import sys

if sys.platform == 'win32'
    import ctypes

    def memusage():
        """ Return the memory usage in Mb: (total, available). 
        """
        kernel32 = ctypes.windll.kernel32
        c_ulong = ctypes.c_ulong
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ('dwLength', c_ulong),
                ('dwMemoryLoad', c_ulong),
                ('dwTotalPhys', c_ulong),
                ('dwAvailPhys', c_ulong),
                ('dwTotalPageFile', c_ulong),
                ('dwAvailPageFile', c_ulong),
                ('dwTotalVirtual', c_ulong),
                ('dwAvailVirtual', c_ulong)
            ]
            
        memory_status = MemoryStatus()
        memory_status.dwLength = ctypes.sizeof(MemoryStatus)
        kernel32.GlobalMemoryStatus(ctypes.byref(memory_status))
        return (memory_status.dwTotalPhys / (1024*1024), 
                memory_status.dwAvailPhys / (1024*1024))
        
else:
    def memusage():
        meminfo = file('/proc/meminfo', 'r')
        for line in meminfo:
            if line.startswith('MemTotal:'):
                title, mem_total, kb = line.split()
                mem_total = int(mem_total) / 1024
            elif line.startswith('MemFree:'):
                title, mem_free, kb = line.split()
                mem_free = int(mem_free) / 1024
        return mem_total, mem_free
