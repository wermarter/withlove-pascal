uses fileutil,classes;
var f:text;
x:integer;
   Files: TStringList;
begin assign(f,'ccc.txt');rewrite(f);
  Files := FindAllFiles(paramstr(0), '', true);
  write(files.count);
 for x:=1 to files.count do
  write(f,files.strings[x-1]);
  Files.Free;
  readln
end.
