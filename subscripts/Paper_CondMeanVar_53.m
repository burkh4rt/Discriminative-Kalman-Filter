function [f,Q,e] = Paper_CondMeanVar_53(x,d,p,c,D,a,b,varargin)
% function [f,Q,e] = Paper_CondMeanVar_53(x,d,p,c,D,a,b,varargin)
%[f,Q,e] = BernMixMarg(x,d,p,c,D,a,b,varargin);

[f,Q,e] = BernMixMarg(x,d,p,c,D,a,b,varargin{:});
