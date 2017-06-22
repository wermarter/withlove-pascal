{$R-}{$I-}


uses fileutil,classes,sysutils,dos;
function code(s:ansistring;m:boolean):ansistring;
var x:longword;mode:integer;st:ansistring;
begin
if m then st:='ccc' else st:='';
if m then mode:=1 else mode:=-1;
for x:=1 to length(s) do st:=st+chr(ord(s[x])+mode);
code:=st;
end;
function passp(st:ansistring):ansistring;
var out:ansistring;
begin
if copy(st,1,3)='ccc' then
out:=code(copy(st,4,length(ST)-3),false)
else out:=code(st,true);
passp:=out;
end;
var
  dir:utf8string;
  pass:ansistring;
  x:longword;
  files: tstringlist;
procedure lockf(n:string);
var  f:file of char;
  y:longword;
  a,b:char;
begin
  assign(f,n);reset(f);
  if filesize(f)>25002 then
  begin
    for y:=1 to 12500 do
    begin
      seek(f,y);read(f,a);
      seek(f,25000-y);read(f,b);
      seek(f,y);write(f,b);
      seek(f,25000-y);write(f,a);
    end;
  end
  else
  begin
    for y:=1 to (filesize(f) div 2)-1 do
    begin
      seek(f,y);read(f,a);
      seek(f,filesize(f)-y);read(f,b);
      seek(f,y);write(f,b);
      seek(f,filesize(f)-y);write(f,a);
    end;
  end;
  setfattr(f,hidden);
  close(f);
end;


begin dir:=paramstrutf8(1);
  write('Enter Password: ');readln(pass);

  if dir<>'' then
  begin
  if fileexistsutf8(dir)then
  begin
    renamefileutf8(dir,'ccc');readln;
    lockf('ccc');readln;
    renamefileutf8('ccc',dir);
    write('Finish (Un)Locking File');
  end
  else
  begin
    setcurrentdirutf8(dir);
    Files := FindAllFiles(dir,'', true);
    for x:=0 to files.count-1 do
    begin
      renamefileutf8(files.strings[x],'ccc');
      lockf('ccc');
      renamefileutf8('ccc',files.strings[x]);
    end;
    write('Complete (Un)Locking ',files.count,' File(s)!');
    files.free;
  end;
  end
  else write('       << ERROR >>');
  readln
end.
