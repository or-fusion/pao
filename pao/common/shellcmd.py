
import sys
import subprocess
import io

from pyomo.common.collections import Bunch
from pyomo.common.tee import TeeStream


def run_shellcmd(cmd, *, env=None, tee=False, time_limit=None):
    ostr = io.StringIO()
    if not tee:
        rc, log = subprocess.run(cmd, env=env, timeout=time_limit,
                stdout=ostr,
                stderr=ostr)
    else:
        with TeeStream(*ostr) as t:
            results = subprocess.run(
                cmd,
                env=env,
                stdout=t.STDOUT,
                stderr=t.STDERR,
                timeout=time_limit,
                universal_newlines=True,
            )
        rc = results.returncode
        log = ostr[0].getvalue()

    return Bunch(rc=rc, log=ostr.getvalue())

