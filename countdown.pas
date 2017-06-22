uses sysutils,crt;
var h,m,s,ms,x:word;
d:real;t:qword;
begin
d:=encodedate(2015,2,19)+encodetime(0,0,0,0)-now;
decodetime(d,h,m,s,ms);
t:=ms+1000*s+1000*60*m+1000*60*60*h;t:=t*1000;
for x:=t downto 0 do
begin gotoxy(2,2);
write(x div 3600,' Hour(s) ',(x mod 3600)div 60,' Minute(s) ',((x mod 3600)mod 60)div 60,' Second(s) Left...');delay(999);
end;

readln
end.
