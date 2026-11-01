import sys,subprocess,numpy as np
f=sys.argv[1]
raw=subprocess.run(['ffmpeg','-v','error','-i',f,'-vf','fps=2,extractplanes=y','-f','rawvideo','-pix_fmt','gray','-'],capture_output=True).stdout
s=(np.frombuffer(raw,np.uint8).astype(float)-16)/219
print('mean %.2f median %.2f >75%% %.1f%% >90%% %.1f%%'%(s.mean(),np.median(s),(s>.75).mean()*100,(s>.9).mean()*100))
