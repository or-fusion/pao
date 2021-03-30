
import sys
import subprocess
import io

from pyutilib.misc import Bunch
from pyutilib.subprocess import run_command



def run_shellcmd(cmd, *, env=None, tee=False, time_limit=None):
    ostr = io.StringIO()
    rc, log = run_command(cmd, tee=tee, env=env, timelimit=time_limit,
            stdout=ostr,
            stderr=ostr)

    return Bunch(rc=rc, log=ostr.getvalue())

