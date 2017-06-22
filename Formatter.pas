uses dos;
var fn:string;
begin
write('FileName: ');readln(fn);
exec('C:\fpc\2.6.4\bin\i386-win32\PascalFormat.exe',fn+'.pas');
end.