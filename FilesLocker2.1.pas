 {$R-}{$I-}


uses fileutil,classes,sysutils;
var
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
  close(f);
end;


begin

  if fileexists(paramstrutf8(1))then
  begin
    renamefileutf8(paramstrutf8(1),'ccc');
    lockf('ccc');
    renamefileutf8('ccc',paramstrutf8(1));
    write('Finish (Un)Locking File');
  end
  else
  begin
    setcurrentdirutf8(paramstrutf8(1));
    Files := FindAllFiles(paramstrutf8(1),'', true);
    for x:=0 to files.count-1 do
    begin
      renamefileutf8(files.strings[x],'ccc');
      lockf('ccc');
      renamefileutf8('ccc',files.strings[x]);
    end;
    write('Complete (Un)Locking ',files.count,' File(s)!');
    files.free;
  end;
  readln
end.
